import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { AIMessage } from '@langchain/core/messages';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import type { DecisionQuestion } from './index';
import {
  createStructuredChatDecisionModel,
  booleanQuestion,
  choiceQuestion,
  scoreQuestion,
} from './index';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { readChatUsage } from './structuredChat';

type ProviderCall = {
  response_format?: {
    json_schema?: { strict?: boolean; schema?: { properties?: object } };
  };
  tools?: Array<{ strict?: boolean; input_schema?: object }>;
  messages?: Array<{ role: string; content: string }>;
};

function openAIModel(
  parsed: object,
  calls: ProviderCall[],
  includeUsage = true,
  method: 'jsonSchema' | 'functionCalling' = 'jsonSchema'
): ChatOpenAI {
  const toolCall = {
    id: 'call_test',
    type: 'function',
    function: { name: 'DecideQuestions', arguments: JSON.stringify(parsed) },
  };
  const fetch: typeof globalThis.fetch = async (_url, init) => {
    calls.push(JSON.parse(String(init?.body)) as ProviderCall);
    return new Response(
      JSON.stringify({
        id: 'chatcmpl-test',
        object: 'chat.completion',
        created: 0,
        model: 'gpt-4o-mini',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: method === 'jsonSchema' ? JSON.stringify(parsed) : null,
              tool_calls: method === 'functionCalling' ? [toolCall] : undefined,
            },
            finish_reason: method === 'functionCalling' ? 'tool_calls' : 'stop',
            logprobs: null,
          },
        ],
        usage: includeUsage
          ? { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 }
          : undefined,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } }
    );
  };
  return new ChatOpenAI({
    model: 'gpt-4o-mini',
    apiKey: 'test',
    maxRetries: 0,
    configuration: { fetch },
  });
}

describe('strict structured-chat decisionModel', () => {
  it.each(['jsonSchema', 'functionCalling'] as const)(
    'batches boolean and choice questions into strict OpenAI %s with raw usage',
    async (method) => {
      const calls: ProviderCall[] = [];
      const model = openAIModel(
        { answers: { q: { decision: true }, role: { choice: 'hook' } } },
        calls,
        true,
        method
      );
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId: 'gpt-4o-mini',
        method,
      });
      const result = await decisionModel.decide({
        state: { text: 'call a test hook' },
        questions: {
          q: booleanQuestion('Is this a test?'),
          role: choiceQuestion('Which role?', {
            hook: 'a hook',
            util: 'a utility',
          }),
        },
      });
      expect(calls).toHaveLength(1);
      const schemaPath =
        method === 'jsonSchema'
          ? 'response_format.json_schema.schema'
          : 'tools.0.function.parameters';
      expect(calls[0]).toHaveProperty(
        method === 'jsonSchema'
          ? 'response_format.json_schema.strict'
          : 'tools.0.function.strict',
        true
      );
      expect(calls[0]).toHaveProperty(
        `${schemaPath}.properties.answers.properties.q.properties.decision.type`,
        'boolean'
      );
      expect(calls[0]).toHaveProperty(
        `${schemaPath}.properties.answers.properties.role.properties.choice.enum`,
        ['hook', 'util']
      );
      expect(calls[0].messages?.[1].content).toContain('call a test hook');
      expect(calls[0].messages?.[1].content).toMatch(
        /^librechat-decision-state:/
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
      expect(result.usage).toEqual({ inputTokens: 12, outputTokens: 4 });
    }
  );

  it('uses actual Anthropic strict tool calling, not JSON-mode fallback', async () => {
    const calls: ProviderCall[] = [];
    const fetch: typeof globalThis.fetch = async (_url, init) => {
      calls.push(JSON.parse(String(init?.body)) as ProviderCall);
      return new Response(
        JSON.stringify({
          id: 'msg_test',
          type: 'message',
          role: 'assistant',
          model: 'claude-haiku-4-5-20251001',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_test',
              name: 'DecideQuestions',
              input: { answers: { q: { decision: false } } },
            },
          ],
          stop_reason: 'tool_use',
          stop_sequence: null,
          usage: { input_tokens: 8, output_tokens: 2 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } }
      );
    };
    const model = new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      anthropicApiKey: 'testing',
      clientOptions: { fetch },
      maxRetries: 0,
    });
    const decisionModel = createStructuredChatDecisionModel({
      model,
      modelId: 'claude-haiku-4-5-20251001',
      method: 'functionCalling',
    });
    const result = await decisionModel.decide({
      state: 'test',
      questions: { q: booleanQuestion('?') },
    });
    expect(calls).toHaveLength(1);
    expect(calls[0].tools?.[0]).toHaveProperty('strict', true);
    expect(calls[0].tools?.[0]).toHaveProperty(
      'input_schema.properties.answers.properties.q.properties.decision.type',
      'boolean'
    );
    expect(result.answers.q).toEqual({
      type: 'boolean',
      probability: null,
      decision: false,
    });
    expect(result.usage).toEqual({ inputTokens: 8, outputTokens: 2 });
  });

  it.each([
    [400, 'bad_request'],
    [401, 'unauthorized'],
    [403, 'unauthorized'],
    [429, 'rate_limited'],
    [503, 'server_error'],
  ] as const)(
    'keeps provider HTTP %s as %s without leaking its body',
    async (status, failure) => {
      const fetch: typeof globalThis.fetch = async () =>
        new Response(
          JSON.stringify({ error: { message: 'private provider response' } }),
          { status, headers: { 'retry-after': '2' } }
        );
      const model = new ChatOpenAI({
        model: 'gpt-4o-mini',
        apiKey: 'test',
        maxRetries: 0,
        configuration: { fetch },
      });
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
      });
      try {
        await decisionModel.decide({
          state: 'private state',
          questions: { q: booleanQuestion('?') },
        });
        throw new Error('expected a classified provider failure');
      } catch (error) {
        expect(error).toMatchObject({ failure, status });
        expect(String(error)).not.toMatch(
          /private provider response|private state/
        );
        if (status === 429) {
          expect(error).toMatchObject({ retryAfterMs: 2000 });
        }
      }
    }
  );

  it('preserves Anthropic rate-limit failures after its provider adapter retries', async () => {
    const fetch: typeof globalThis.fetch = async () =>
      new Response(
        JSON.stringify({
          type: 'error',
          error: { type: 'rate_limit_error', message: 'private response' },
        }),
        { status: 429, headers: { 'retry-after': '3' } }
      );
    const model = new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      anthropicApiKey: 'test',
      clientOptions: { fetch },
      maxRetries: 0,
    });
    const decisionModel = createStructuredChatDecisionModel({
      model,
      modelId: 'claude-haiku-4-5-20251001',
      method: 'functionCalling',
    });
    await expect(
      decisionModel.decide({
        state: 'test',
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({
      failure: 'rate_limited',
      status: 429,
      retryAfterMs: 3000,
    });
  });

  it.each([13, 1025])(
    'counts Bedrock-shaped cache buckets once from raw input usage %s',
    (inputTokens) => {
      const raw = new AIMessage({
        content: '',
        usage_metadata: {
          input_tokens: inputTokens,
          output_tokens: 5,
          total_tokens: 1030,
        },
        response_metadata: {
          usage: {
            inputTokens: 13,
            outputTokens: 5,
            cacheReadInputTokens: 1000,
            cacheWriteInputTokens: 12,
          },
        },
      });
      expect(readChatUsage(raw)).toEqual({
        inputTokens: 1025,
        outputTokens: 5,
      });
    }
  );

  it.each(['jsonSchema', 'functionCalling'] as const)(
    'rejects Bedrock %s before invocation even when its adapter silently accepts strict options',
    async (method) => {
      const model = new CustomChatBedrockConverse({
        model: 'anthropic.claude-3-sonnet-20240229-v1:0',
        region: 'us-east-1',
        credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
      });
      const generate = jest.spyOn(model, '_generate').mockResolvedValue({
        generations: [
          {
            text: '',
            message: new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'tool-test',
                  name: 'DecideQuestions',
                  args: { answers: { q: { decision: true } } },
                },
              ],
            }),
          },
        ],
      });
      const structured = jest.spyOn(model, 'withStructuredOutput');
      const onAnswered = jest.fn();
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId: 'gpt-4o-mini',
        providerId: 'openai',
        method,
        onAnswered,
      });
      await expect(
        decisionModel.decide({
          state: 'test',
          questions: { q: booleanQuestion('?') },
        })
      ).rejects.toMatchObject({ failure: 'unsupported_mode' });
      expect(structured).not.toHaveBeenCalled();
      expect(generate).not.toHaveBeenCalled();
      expect(onAnswered).not.toHaveBeenCalled();
    }
  );

  it('rejects unverified adapters even when their structured pipeline would return valid JSON', async () => {
    const model = new ChatGoogleGenerativeAI({
      model: 'gemini-2.5-flash',
      apiKey: 'test',
    });
    const generate = jest.spyOn(model, '_generate').mockResolvedValue({
      generations: [
        {
          text: '',
          message: new AIMessage({
            content: JSON.stringify({ answers: { q: { decision: true } } }),
          }),
        },
      ],
    });
    const structured = jest.spyOn(model, 'withStructuredOutput');
    const decisionModel = createStructuredChatDecisionModel({
      model,
      modelId: model.model,
      method: 'jsonSchema',
    });
    await expect(
      decisionModel.decide({
        state: 'test',
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({ failure: 'unsupported_mode' });
    expect(structured).not.toHaveBeenCalled();
    expect(generate).not.toHaveBeenCalled();
  });

  it('rejects an unsupported provider mode and rejects score before invocation', async () => {
    const calls: ProviderCall[] = [];
    const model = openAIModel({ answers: { q: { decision: true } } }, calls);
    const legacy = new ChatOpenAI({
      model: 'gpt-4',
      apiKey: 'testing',
      maxRetries: 0,
    });
    const notSupported = createStructuredChatDecisionModel({
      model: legacy,
      modelId: 'gpt-4',
      method: 'jsonSchema',
    });
    await expect(
      notSupported.decide({
        state: {},
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({ failure: 'unsupported_mode' });
    expect(calls).toHaveLength(0);

    const chat = createStructuredChatDecisionModel({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    await expect(
      chat.decide({
        state: {},
        questions: { score: scoreQuestion('How bad?', ['fine', 'bad']) },
      })
    ).rejects.toMatchObject({ failure: 'unsupported_question' });
    const tooMany = Object.fromEntries(
      Array.from({ length: 33 }, (_, index) => [
        `q${index}`,
        booleanQuestion('?'),
      ])
    );
    await expect(
      chat.decide({ state: {}, questions: tooMany })
    ).rejects.toMatchObject({ failure: 'unsupported_question' });
    expect(calls).toHaveLength(0);
  });

  it.each([
    { answers: { q: { decision: 'true' } } },
    { answers: { q: { decision: true, probability: 0.99 } } },
    { answers: { q: { decision: true }, extra: { decision: false } } },
    { answers: {} },
    { answers: { q: { choice: 'yes' } } },
  ])(
    'rejects malformed or invented decisions even when provider parsing succeeds',
    async (parsed) => {
      const calls: ProviderCall[] = [];
      const onAnswered = jest.fn();
      const decisionModel = createStructuredChatDecisionModel({
        model: openAIModel(parsed, calls),
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
        onAnswered,
      });
      await expect(
        decisionModel.decide({
          state: 'secret input',
          questions: { q: booleanQuestion('?') },
        })
      ).rejects.toMatchObject({ failure: 'malformed_response' });
      expect(calls).toHaveLength(1);
      expect(onAnswered).not.toHaveBeenCalled();
    }
  );

  it.each([
    null,
    [],
    7,
    true,
    false,
    { yes: 'Yes' },
    { true: 7 },
    { false: null },
  ])(
    'rejects malformed boolean criteria %j before chat invocation',
    async (criteria) => {
      const calls: ProviderCall[] = [];
      const decisionModel = createStructuredChatDecisionModel({
        model: openAIModel({ answers: { q: { decision: true } } }, calls),
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
      });
      const question = JSON.parse(
        JSON.stringify({ type: 'boolean', instructions: '?', criteria })
      ) as DecisionQuestion;
      await expect(
        decisionModel.decide({ state: {}, questions: { q: question } })
      ).rejects.toMatchObject({ failure: 'bad_request' });
      expect(calls).toHaveLength(0);
    }
  );

  it('preserves unknown usage and validates choices even if the model returns another label', async () => {
    const calls: ProviderCall[] = [];
    const decisionModel = createStructuredChatDecisionModel({
      model: openAIModel({ answers: { q: { decision: false } } }, calls, false),
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const result = await decisionModel.decide({
      state: 'test',
      questions: { q: booleanQuestion('?') },
    });
    expect(result.usage).toBeNull();

    const invalid = createStructuredChatDecisionModel({
      model: openAIModel(
        { answers: { choice: { choice: 'not listed' } } },
        calls
      ),
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    await expect(
      invalid.decide({
        state: 'test',
        questions: { choice: choiceQuestion('Which?', { a: 'A', b: 'B' }) },
      })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
  });

  it('validates a delayed answer against the question sent before caller mutation', async () => {
    let reply: ((response: Response) => void) | undefined;
    let requested: (() => void) | undefined;
    const sent = new Promise<void>((resolve) => {
      requested = resolve;
    });
    const pending = new Promise<Response>((resolve) => {
      reply = resolve;
    });
    let payload: ProviderCall | undefined;
    const fetch: typeof globalThis.fetch = async (_url, init) => {
      payload = JSON.parse(String(init?.body)) as ProviderCall;
      requested?.();
      return pending;
    };
    const model = new ChatOpenAI({
      model: 'gpt-4o-mini',
      apiKey: 'test',
      maxRetries: 0,
      configuration: { fetch },
    });
    const decisionModel = createStructuredChatDecisionModel({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const pick = choiceQuestion('Pick?', { first: 'First', second: 'Second' });
    const resultPromise = decisionModel.decide({
      state: 'hello',
      questions: { pick },
    });
    await sent;
    delete pick.criteria.first;
    pick.criteria.third = 'Third';
    reply?.(
      new Response(
        JSON.stringify({
          id: 'chatcmpl-test',
          object: 'chat.completion',
          created: 0,
          model: 'gpt-4o-mini',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: JSON.stringify({
                  answers: { pick: { choice: 'first' } },
                }),
              },
              finish_reason: 'stop',
            },
          ],
        }),
        { status: 200, headers: { 'content-type': 'application/json' } }
      )
    );
    expect((await resultPromise).answers.pick).toMatchObject({
      type: 'choice',
      choice: 'first',
    });
    expect(payload?.messages?.[1].content).toContain('"first":"First"');
    expect(payload).toHaveProperty(
      'response_format.json_schema.schema.properties.answers.properties.pick.properties.choice.enum',
      ['first', 'second']
    );
  });

  it.each(['question', 'criterion'] as const)(
    'sanitizes a throwing %s getter before provider invocation',
    async (source) => {
      const calls: ProviderCall[] = [];
      const model = openAIModel({ answers: { q: { decision: true } } }, calls);
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
      });
      const fail = (): never => {
        throw new Error('private question text and API key');
      };
      const dynamicQuestions = {
        get q(): DecisionQuestion {
          return fail();
        },
      };
      const dynamicCriteria = {
        get a(): string {
          return fail();
        },
        b: 'B',
      };
      const choiceQuestions = { q: choiceQuestion('Which?', dynamicCriteria) };
      const questions =
        source === 'question' ? dynamicQuestions : choiceQuestions;
      try {
        await decisionModel.decide({ state: 'test', questions });
        throw new Error('expected sanitized preparation failure');
      } catch (error) {
        expect(error).toMatchObject({
          failure: 'bad_request',
          provider: 'structured-chat',
        });
        expect(String(error)).not.toContain('private');
      }
      expect(calls).toHaveLength(0);
    }
  );

  it('checks a pre-aborted signal before reading dynamic questions', async () => {
    const model = openAIModel({ answers: { q: { decision: true } } }, []);
    const decisionModel = createStructuredChatDecisionModel({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const controller = new AbortController();
    controller.abort();
    const questions = Object.defineProperty({}, 'q', {
      enumerable: true,
      get: () => {
        throw new Error('aborted questions were read');
      },
    }) as Record<string, ReturnType<typeof booleanQuestion>>;
    await expect(
      decisionModel.decide({
        state: {},
        questions,
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ failure: 'aborted' });
  });

  it('honors per-call aborts and a provider request deadline', async () => {
    const fetch: typeof globalThis.fetch = async (_url, init) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        );
      });
    const model = new ChatOpenAI({
      model: 'gpt-4o-mini',
      apiKey: 'testing',
      maxRetries: 0,
      configuration: { fetch },
    });
    const decisionModel = createStructuredChatDecisionModel({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
      timeoutMs: 25,
    });
    const cancelled = new AbortController();
    cancelled.abort();
    await expect(
      decisionModel.decide({
        state: {},
        questions: { q: booleanQuestion('?') },
        signal: cancelled.signal,
      })
    ).rejects.toMatchObject({ failure: 'aborted' });
    await expect(
      decisionModel.decide({
        state: {},
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({ failure: 'timeout' });
  });
});

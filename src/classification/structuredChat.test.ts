import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { AIMessage } from '@langchain/core/messages';
import {
  createStructuredChatClassifier,
  booleanQuestion,
  choiceQuestion,
  scoreQuestion,
} from './index';
import { CustomChatBedrockConverse } from '@/llm/bedrock';

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
  includeUsage = true
): ChatOpenAI {
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
            message: { role: 'assistant', content: JSON.stringify(parsed) },
            finish_reason: 'stop',
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

describe('strict structured-chat classifier', () => {
  it('batches boolean and choice questions into one provider-enforced schema with raw usage', async () => {
    const calls: ProviderCall[] = [];
    const model = openAIModel(
      { answers: { q: { decision: true }, role: { choice: 'hook' } } },
      calls
    );
    const classifier = createStructuredChatClassifier({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const result = await classifier.classify({
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
    expect(calls[0]).toHaveProperty('response_format.json_schema.strict', true);
    expect(calls[0]).toHaveProperty(
      'response_format.json_schema.schema.properties.answers.properties.q.properties.decision.type',
      'boolean'
    );
    expect(calls[0]).toHaveProperty(
      'response_format.json_schema.schema.properties.answers.properties.role.properties.choice.enum',
      ['hook', 'util']
    );
    expect(calls[0].messages?.[1].content).toContain('call a test hook');
    expect(calls[0].messages?.[1].content).toMatch(
      /^librechat-classifier-state:/
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
  });

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
              name: 'ClassifyDecisions',
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
    const classifier = createStructuredChatClassifier({
      model,
      modelId: 'claude-haiku-4-5-20251001',
      method: 'functionCalling',
    });
    const result = await classifier.classify({
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
      const classifier = createStructuredChatClassifier({
        model,
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
      });
      try {
        await classifier.classify({
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
    const classifier = createStructuredChatClassifier({
      model,
      modelId: 'claude-haiku-4-5-20251001',
      method: 'functionCalling',
    });
    await expect(
      classifier.classify({
        state: 'test',
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({
      failure: 'rate_limited',
      status: 429,
      retryAfterMs: 3000,
    });
  });

  it('counts Bedrock prompt cache buckets once, matching the fork billing convention', async () => {
    const model = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-sonnet-20240229-v1:0',
      region: 'us-east-1',
      credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
    });
    let inputTokens = 13;
    const generate = jest
      .spyOn(model, '_generate')
      .mockImplementation(async () => ({
        generations: [
          {
            text: '',
            message: new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'tool-test',
                  name: 'ClassifyDecisions',
                  args: { answers: { q: { decision: true } } },
                },
              ],
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
            }),
          },
        ],
      }));
    const classifier = createStructuredChatClassifier({
      model,
      modelId: 'anthropic.claude-3-sonnet-20240229-v1:0',
      method: 'functionCalling',
    });
    const request = { state: 'test', questions: { q: booleanQuestion('?') } };
    expect((await classifier.classify(request)).usage).toEqual({
      inputTokens: 1025,
      outputTokens: 5,
    });
    inputTokens = 1025;
    expect((await classifier.classify(request)).usage).toEqual({
      inputTokens: 1025,
      outputTokens: 5,
    });
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it('rejects an unsupported provider mode and rejects score before invocation', async () => {
    const calls: ProviderCall[] = [];
    const model = openAIModel({ answers: { q: { decision: true } } }, calls);
    const legacy = new ChatOpenAI({
      model: 'gpt-4',
      apiKey: 'testing',
      maxRetries: 0,
    });
    const notSupported = createStructuredChatClassifier({
      model: legacy,
      modelId: 'gpt-4',
      method: 'jsonSchema',
    });
    await expect(
      notSupported.classify({
        state: {},
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({ failure: 'unsupported_mode' });
    expect(calls).toHaveLength(0);

    const chat = createStructuredChatClassifier({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    await expect(
      chat.classify({
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
      chat.classify({ state: {}, questions: tooMany })
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
      const classifier = createStructuredChatClassifier({
        model: openAIModel(parsed, calls),
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
        onAnswered,
      });
      await expect(
        classifier.classify({
          state: 'secret input',
          questions: { q: booleanQuestion('?') },
        })
      ).rejects.toMatchObject({ failure: 'malformed_response' });
      expect(calls).toHaveLength(1);
      expect(onAnswered).not.toHaveBeenCalled();
    }
  );

  it('preserves unknown usage and validates choices even if the model returns another label', async () => {
    const calls: ProviderCall[] = [];
    const classifier = createStructuredChatClassifier({
      model: openAIModel({ answers: { q: { decision: false } } }, calls, false),
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const result = await classifier.classify({
      state: 'test',
      questions: { q: booleanQuestion('?') },
    });
    expect(result.usage).toBeNull();

    const invalid = createStructuredChatClassifier({
      model: openAIModel(
        { answers: { choice: { choice: 'not listed' } } },
        calls
      ),
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    await expect(
      invalid.classify({
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
    const classifier = createStructuredChatClassifier({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const pick = choiceQuestion('Pick?', { first: 'First', second: 'Second' });
    const resultPromise = classifier.classify({
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

  it('checks a pre-aborted signal before reading dynamic questions', async () => {
    const model = openAIModel({ answers: { q: { decision: true } } }, []);
    const classifier = createStructuredChatClassifier({
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
      classifier.classify({
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
    const classifier = createStructuredChatClassifier({
      model,
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
      timeoutMs: 25,
    });
    const cancelled = new AbortController();
    cancelled.abort();
    await expect(
      classifier.classify({
        state: {},
        questions: { q: booleanQuestion('?') },
        signal: cancelled.signal,
      })
    ).rejects.toMatchObject({ failure: 'aborted' });
    await expect(
      classifier.classify({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'timeout' });
  });
});

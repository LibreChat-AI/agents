import { ChatOpenAICompletions } from '@langchain/openai';
import { getRetryable, ContextOverflowError } from '@langchain/core/errors';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { OpenAIClientOptions } from '@/types/llm';
import {
  ChatOpenAI,
  AzureChatOpenAI,
  CustomOpenAIClient,
  CustomAzureOpenAIClient,
} from './index';
import {
  RequestBodyBudgetError,
  RequestBodyLimitError,
} from '@/llm/requestBodyBudget';

type Route = 'chat' | 'responses';
type Mode = 'invoke' | 'stream' | 'events';

function reply(route: Route, stream: boolean): Response {
  const chat = {
    id: 'chat_fixture',
    object: 'chat.completion',
    created: 1,
    model: 'gpt-4o',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: '{"ok":true}' },
        finish_reason: 'stop',
      },
    ],
  };
  const response = {
    id: 'resp_fixture',
    object: 'response',
    created_at: 1,
    status: 'completed',
    model: 'gpt-4o',
    output: [
      {
        type: 'message',
        id: 'msg_fixture',
        status: 'completed',
        role: 'assistant',
        content: [
          { type: 'output_text', text: '{"ok":true}', annotations: [] },
        ],
      },
    ],
  };
  if (!stream) {
    return Response.json(route === 'chat' ? chat : response);
  }
  const events =
    route === 'chat'
      ? [
        {
          ...chat,
          object: 'chat.completion.chunk',
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'OK' },
              finish_reason: 'stop',
            },
          ],
        },
      ]
      : [{ type: 'response.completed', response, sequence_number: 0 }];
  const body =
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') +
    'data: [DONE]\n\n';
  return new Response(body, {
    headers: { 'content-type': 'text/event-stream' },
  });
}

function modelFor(
  azure: boolean,
  route: Route,
  bodies: string[],
  cap?: number,
  extras: Partial<OpenAIClientOptions> = {}
): ChatOpenAI | AzureChatOpenAI {
  const transport: typeof fetch = async (_url, init) => {
    if (typeof init?.body !== 'string') {
      throw new Error('Fixture expects serialized JSON');
    }
    bodies.push(init.body);
    const request = JSON.parse(init.body) as { stream?: boolean };
    return reply(route, request.stream === true);
  };
  const fields = {
    apiKey: 'offline-key',
    model: 'gpt-4o',
    useResponsesApi: route === 'responses',
    _lc_stream_delay: 0,
    maxTokens: 32,
    promptCacheExplicit: true,
    maxRequestBodyBytes: cap,
    configuration: { baseURL: 'http://synthetic.invalid/v1', fetch: transport },
    ...extras,
  };
  return azure
    ? new AzureChatOpenAI({
      ...fields,
      azureOpenAIApiKey: 'offline-key',
      azureOpenAIApiVersion: '2024-10-21',
      azureOpenAIApiInstanceName: 'fixture',
      azureOpenAIApiDeploymentName: 'fixture',
    })
    : new ChatOpenAI(fields);
}

async function invoke(
  model: ChatOpenAI | AzureChatOpenAI,
  mode: Mode
): Promise<void> {
  const messages = [
    new SystemMessage('Stable instructions'),
    new HumanMessage('Unicode 日本語 🧪 "escaped"\n'),
  ];
  if (mode === 'invoke') {
    await model.invoke(messages);
    return;
  }
  if (mode === 'events') {
    const events = model.streamEvents(messages);
    for await (const event of events) {
      expect(event).toBeDefined();
    }
    await events.output;
    return;
  }
  for await (const chunk of await model.stream(messages)) {
    expect(chunk).toBeDefined();
  }
}

describe.each<[boolean, Route]>([
  [false, 'chat'],
  [false, 'responses'],
  [true, 'chat'],
  [true, 'responses'],
])('serialized body budget: Azure=%s route=%s', (azure, route) => {
  afterEach(() => jest.restoreAllMocks());

  it.each<Mode>(['invoke', 'stream', 'events'])(
    'keeps exact passing bytes and rejects one byte less in %s',
    async (mode) => {
      const baseline: string[] = [];
      await invoke(modelFor(azure, route, baseline), mode);
      expect(baseline).toHaveLength(1);
      const bytes = Buffer.byteLength(baseline[0], 'utf8');
      expect(bytes).toBeGreaterThan(baseline[0].length);
      expect(baseline[0]).toContain('prompt_cache_breakpoint');
      const passing: string[] = [];
      await invoke(modelFor(azure, route, passing, bytes), mode);
      expect(passing).toEqual(baseline);

      const rejected: string[] = [];
      const built = jest.spyOn(
        azure
          ? CustomAzureOpenAIClient.prototype
          : CustomOpenAIClient.prototype,
        'buildRequest'
      );
      let failure: Error | undefined;
      try {
        await invoke(modelFor(azure, route, rejected, bytes - 1), mode);
      } catch (error) {
        if (!(error instanceof Error)) throw error;
        failure = error;
      }
      expect(failure).toBeInstanceOf(RequestBodyLimitError);
      expect(failure).toMatchObject({
        source: 'local_request_body',
        observedBytes: bytes,
        limitBytes: bytes - 1,
      });
      expect(failure).not.toBeInstanceOf(ContextOverflowError);
      expect(getRetryable(failure)).toBe(false);
      expect(built).toHaveBeenCalledTimes(1);
      expect(rejected).toEqual([]);
    }
  );

  it('measures SDK parse helpers and structured output without changing the body', async () => {
    const schema = {
      type: 'object',
      properties: { ok: { type: 'boolean' } },
      required: ['ok'],
      additionalProperties: false,
    };
    const extras =
      route === 'chat'
        ? {
          modelKwargs: {
            response_format: {
              type: 'json_schema',
              json_schema: { name: 'answer', strict: true, schema },
            },
          },
        }
        : {
          modelKwargs: {
            text: {
              format: {
                type: 'json_schema',
                name: 'answer',
                strict: true,
                schema,
              },
            },
          },
        };
    const bodies: string[] = [];
    await invoke(modelFor(azure, route, bodies, undefined, extras), 'invoke');
    expect(bodies[0]).toContain('json_schema');
    const exact = Buffer.byteLength(bodies[0], 'utf8');
    const passing: string[] = [];
    await invoke(modelFor(azure, route, passing, exact, extras), 'invoke');
    expect(passing).toEqual(bodies);
    await expect(
      invoke(modelFor(azure, route, [], exact - 1, extras), 'invoke')
    ).rejects.toBeInstanceOf(RequestBodyLimitError);
  });

  it('checks the effective per-call body and fetchOptions override', async () => {
    const oversized = {
      model: 'gpt-4o',
      input: '🧪'.repeat(500),
      messages: [{ role: 'user', content: '🧪'.repeat(500) }],
    };
    const bodies: string[] = [];
    const model = modelFor(azure, route, bodies, 400);
    await expect(
      model.invoke('short', { options: { body: oversized } })
    ).rejects.toMatchObject({
      observedBytes: Buffer.byteLength(JSON.stringify(oversized)),
      limitBytes: 400,
    });
    await expect(
      model.invoke('short', {
        options: {
          fetchOptions: Object.defineProperty({}, 'body', {
            value: JSON.stringify(oversized),
            enumerable: true,
          }),
        },
      })
    ).rejects.toMatchObject({
      observedBytes: Buffer.byteLength(JSON.stringify(oversized)),
      limitBytes: 400,
    });
    expect(bodies).toEqual([]);
  });
});

describe('request body guard configuration', () => {
  it.each([0, -1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid limit %s',
    (limit) => {
      expect(() => modelFor(false, 'chat', [], limit)).toThrow(RangeError);
      expect(() => modelFor(true, 'responses', [], limit)).toThrow(RangeError);
    }
  );

  it('rejects unsupported non-string bodies before fetch without consuming them', async () => {
    const bodies: string[] = [];
    const model = modelFor(false, 'responses', bodies, 1000);
    const body = new Blob(['small']);
    await expect(
      model.invoke('short', {
        options: {
          fetchOptions: Object.defineProperty({}, 'body', {
            value: body,
            enumerable: true,
          }),
        },
      })
    ).rejects.toBeInstanceOf(RequestBodyBudgetError);
    expect(bodies).toEqual([]);
    expect(body.size).toBe(5);
  });

  it('rejects replacement delegates only when the limit is enabled', () => {
    const delegate = new ChatOpenAICompletions({ apiKey: 'offline-key' });
    expect(
      () => new ChatOpenAI({ apiKey: 'offline-key', completions: delegate })
    ).not.toThrow();
    expect(
      () =>
        new ChatOpenAI({
          apiKey: 'offline-key',
          completions: delegate,
          maxRequestBodyBytes: 1000,
        })
    ).toThrow(RequestBodyBudgetError);
  });

  it('does not let a host retry handler retry a local rejection', async () => {
    const handler = jest.fn(() => undefined);
    const built = jest.spyOn(CustomOpenAIClient.prototype, 'buildRequest');
    const bodies: string[] = [];
    try {
      await expect(
        invoke(
          modelFor(false, 'responses', bodies, 1, {
            maxRetries: 5,
            onFailedAttempt: handler,
          }),
          'invoke'
        )
      ).rejects.toBeInstanceOf(RequestBodyLimitError);
      expect(built).toHaveBeenCalledTimes(1);
      expect(handler).not.toHaveBeenCalled();
      expect(bodies).toEqual([]);
    } finally {
      built.mockRestore();
    }
  });

  it('rejects SDK provider hooks that can mutate a body after buildRequest', () => {
    const options = Object.defineProperty(
      { apiKey: 'offline-key', maxRequestBodyBytes: 1000 },
      'provider',
      { value: {}, enumerable: true }
    );
    expect(() => new CustomOpenAIClient(options)).toThrow(
      RequestBodyBudgetError
    );
  });
});

import type { ClassificationFetch } from './index';
import {
  createClassifier,
  createHttpClassifier,
  classificationPreset,
  mergeClassificationSettings,
  ClassificationError,
  booleanQuestion,
  choiceQuestion,
  scoreQuestion,
  toWireQuestion,
  readAnswer,
  parseEnvelope,
} from './index';

type FetchCall = {
  url: string;
  body: Record<string, unknown>;
  auth: string | undefined;
};

function fakeFetch(
  responses: Array<{ status: number; body: string; retryAfter?: string }>,
  calls: FetchCall[] = []
): { fetch: ClassificationFetch; calls: FetchCall[] } {
  let i = 0;
  const fetch: ClassificationFetch = async (url, init) => {
    calls.push({
      url,
      body: JSON.parse(init.body) as Record<string, unknown>,
      auth: init.headers.Authorization,
    });
    const r = responses[Math.min(i, responses.length - 1)];
    i += 1;
    return {
      ok: r.status >= 200 && r.status < 300,
      status: r.status,
      headers: {
        get: (name: string) =>
          name === 'retry-after' ? (r.retryAfter ?? null) : null,
      },
      text: async () => r.body,
    };
  };
  return { fetch, calls };
}

describe('classification dialect', () => {
  it('sends a boolean as a noul with a {true} pair on System One, unchanged on the port', () => {
    expect(
      toWireQuestion(booleanQuestion('Is it a test?'), 'systemone')
    ).toEqual({ type: 'noul', instructions: 'Is it a test?' });
    expect(
      toWireQuestion(
        booleanQuestion('Is it a test?', 'it calls test()'),
        'systemone'
      )
    ).toEqual({
      type: 'noul',
      instructions: 'Is it a test?',
      criteria: { true: 'it calls test()' },
    });
    expect(
      toWireQuestion(
        booleanQuestion('Is it a test?', {
          true: 'yes when',
          false: 'no when',
        }),
        'systemone'
      )
    ).toEqual({
      type: 'noul',
      instructions: 'Is it a test?',
      criteria: { true: 'yes when', false: 'no when' },
    });
    expect(
      toWireQuestion(
        booleanQuestion('Is it a test?', 'it calls test()'),
        'port'
      )
    ).toEqual({
      type: 'boolean',
      instructions: 'Is it a test?',
      criteria: { true: 'it calls test()' },
    });
    expect(
      toWireQuestion(choiceQuestion('Which?', { a: 'A', b: null }), 'systemone')
    ).toEqual({
      type: 'choice',
      instructions: 'Which?',
      criteria: { a: 'A', b: null },
    });
    expect(
      toWireQuestion(scoreQuestion('How bad?', ['fine', 'bad']), 'systemone')
    ).toEqual({
      type: 'score',
      instructions: 'How bad?',
      criteria: ['fine', 'bad'],
    });
  });

  it('reads answers back in the port shape and drops what it cannot read', () => {
    expect(readAnswer({ type: 'noul', noul: 0.83 }, 'systemone')).toEqual({
      type: 'boolean',
      probability: 0.83,
    });
    expect(readAnswer({ type: 'boolean', probability: 0.2 }, 'port')).toEqual({
      type: 'boolean',
      probability: 0.2,
    });
    expect(
      readAnswer(
        {
          type: 'choice',
          choice: 'hook',
          confidence: 0.91,
          probabilities: { hook: 0.91, util: 0.09 },
        },
        'systemone'
      )
    ).toEqual({
      type: 'choice',
      choice: 'hook',
      confidence: 0.91,
      probabilities: { hook: 0.91, util: 0.09 },
    });
    expect(readAnswer({ type: 'choice', choice: 'hook' }, 'systemone')).toEqual(
      { type: 'choice', choice: 'hook', confidence: null, probabilities: {} }
    );
    expect(
      readAnswer(
        {
          type: 'score',
          score: 1.7,
          confidence: 0.9,
          probabilities: { '0': 0.1, '1': 0.1, '2': 0.8 },
          legend: { '0': 'a' },
        },
        'systemone'
      )
    ).toEqual({
      type: 'score',
      score: 1.7,
      confidence: 0.9,
      probabilities: { '0': 0.1, '1': 0.1, '2': 0.8 },
    });
    expect(readAnswer({ type: 'choice', choice: 7 }, 'systemone')).toBeNull();
    expect(readAnswer('nope', 'systemone')).toBeNull();
  });

  it('parses the envelope and refuses one without answers', () => {
    const env = parseEnvelope(
      '{"model":"jev-1.13.0","answers":{"a":{"type":"noul","noul":0.5},"b":{"type":"weird"}},"usage":{"input_tokens":312,"output_tokens":48}}',
      'systemone',
      (a) => readAnswer(a, 'systemone')
    );
    expect(env).toEqual({
      model: 'jev-1.13.0',
      answers: { a: { type: 'boolean', probability: 0.5 } },
      usage: { inputTokens: 312, outputTokens: 48 },
    });
    expect(() => parseEnvelope('not json', 'x', () => null)).toThrow(
      ClassificationError
    );
    expect(() => parseEnvelope('{"model":"m"}', 'x', () => null)).toThrow(
      /no answers/
    );
  });
});

describe('http classifier', () => {
  it('posts the System One body with the model and reads the reply', async () => {
    const { fetch, calls } = fakeFetch([
      {
        status: 200,
        body: JSON.stringify({
          model: 'jev-1.13.0',
          answers: {
            role: {
              type: 'choice',
              choice: 'hook',
              confidence: 0.7,
              probabilities: { hook: 0.7, util: 0.3 },
            },
            is_test: { type: 'noul', noul: 0.02 },
          },
          usage: { input_tokens: 10, output_tokens: 1 },
        }),
      },
    ]);
    const classifier = createHttpClassifier({
      endpoint: 'https://s1.example/v1/systemone',
      apiKey: 'k',
      model: 'jev-latest',
      dialect: 'systemone',
      fetch,
    });
    const result = await classifier.classify({
      state: { path: 'x.ts' },
      questions: {
        role: choiceQuestion('Which?', { hook: 'a hook', util: 'a util' }),
        is_test: booleanQuestion('test?'),
      },
    });
    expect(calls).toHaveLength(1);
    expect(calls[0].auth).toBe('Bearer k');
    expect(calls[0].body).toEqual({
      model: 'jev-latest',
      state: { path: 'x.ts' },
      questions: {
        role: {
          type: 'choice',
          instructions: 'Which?',
          criteria: { hook: 'a hook', util: 'a util' },
        },
        is_test: { type: 'noul', instructions: 'test?' },
      },
    });
    expect(result.answers.role).toEqual({
      type: 'choice',
      choice: 'hook',
      confidence: 0.7,
      probabilities: { hook: 0.7, util: 0.3 },
    });
    expect(result.answers.is_test).toEqual({
      type: 'boolean',
      probability: 0.02,
    });
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 1 });
  });

  it('wraps the request and unwraps the response for hosts that nest them', async () => {
    const { fetch, calls } = fakeFetch([
      {
        status: 200,
        body: JSON.stringify({
          result: {
            model: 'jev',
            answers: { q: { type: 'noul', noul: 0.9 } },
            usage: {},
          },
        }),
      },
    ]);
    const classifier = createClassifier(
      mergeClassificationSettings(classificationPreset('cloudflare'), {
        baseURL: 'https://cf.example/run',
      }),
      'k',
      { fetch }
    );
    const result = await classifier.classify({
      state: 'hello',
      questions: { q: booleanQuestion('?') },
    });
    expect(calls[0].body).toEqual({
      model: 'typesafe/jev',
      input: {
        state: 'hello',
        questions: { q: { type: 'noul', instructions: '?' } },
      },
    });
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(result.usage).toEqual({ inputTokens: 0, outputTokens: 0 });
  });

  it('does not retry an authorization failure, and retries a rate limit once within the deadline', async () => {
    const denied = fakeFetch([{ status: 403, body: '{"error":"no"}' }]);
    const classifier = createHttpClassifier({
      endpoint: 'https://s1.example/v1/systemone',
      apiKey: 'k',
      fetch: denied.fetch,
    });
    await expect(
      classifier.classify({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'unauthorized', status: 403 });
    expect(denied.calls).toHaveLength(1);

    const limited = fakeFetch([
      { status: 429, body: 'slow down', retryAfter: '0' },
      {
        status: 200,
        body: JSON.stringify({
          model: 'jev',
          answers: { q: { type: 'boolean', probability: 0.4 } },
          usage: {},
        }),
      },
    ]);
    const patient = createHttpClassifier({
      endpoint: 'https://s1.example/v1/systemone',
      apiKey: 'k',
      fetch: limited.fetch,
      sleep: async () => {},
    });
    const result = await patient.classify({
      state: {},
      questions: { q: booleanQuestion('?') },
    });
    expect(limited.calls).toHaveLength(2);
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.4 });
  });

  it('mints a credential function per request and re-mints once after a 401', async () => {
    let minted = 0;
    const seen: Array<{ auth: string | undefined; refresh: boolean }> = [];
    const credential = async ({ refresh }: { refresh: boolean }) => {
      minted += 1;
      seen.push({ auth: undefined, refresh });
      return `tok-${minted}`;
    };
    const ok = JSON.stringify({
      model: 'jev',
      answers: { q: { type: 'noul', noul: 0.9 } },
      usage: {},
    });
    const expiring = fakeFetch([
      { status: 401, body: 'expired' },
      { status: 200, body: ok },
    ]);
    const classifier = createHttpClassifier({
      endpoint: 'https://gw.example/v1/systemone',
      apiKey: credential,
      dialect: 'systemone',
      fetch: expiring.fetch,
      sleep: async () => {},
    });
    const result = await classifier.classify({
      state: {},
      questions: { q: booleanQuestion('?') },
    });
    expect(expiring.calls.map((c) => c.auth)).toEqual([
      'Bearer tok-1',
      'Bearer tok-2',
    ]);
    expect(seen.map((s) => s.refresh)).toEqual([false, true]);
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.9 });

    const forbidden = fakeFetch([{ status: 403, body: 'not this route' }]);
    const scoped = createHttpClassifier({
      endpoint: 'https://gw.example/v1/systemone',
      apiKey: credential,
      fetch: forbidden.fetch,
    });
    await expect(
      scoped.classify({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'unauthorized', status: 403 });
    expect(minted).toBe(3);
    expect(classificationPreset('clickhouse')).toMatchObject({
      baseURL: 'https://inference-internal.clickhouse.cloud/v1/systemone',
      dialect: 'systemone',
      apiKeyEnv: 'CHAI_AUTH_TOKEN',
    });
  });

  it('refuses to start without a key or an endpoint', () => {
    expect(() =>
      createHttpClassifier({ endpoint: 'https://s1.example', apiKey: '' })
    ).toThrow(/API key/);
    expect(() => createHttpClassifier({ endpoint: '', apiKey: 'k' })).toThrow(
      /endpoint/
    );
    expect(classificationPreset('typesafe')).toMatchObject({
      baseURL: 'https://api.typesafe.ai/v1/systemone',
      model: 'jev-latest',
      dialect: 'systemone',
    });
    expect(classificationPreset('nope')).toBeNull();
  });
});

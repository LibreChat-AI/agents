import type { DecisionFetch, DecisionQuestion } from './index';
import {
  createDecisionModel,
  createHttpDecisionModel,
  decisionPreset,
  DECISION_PRESETS,
  mergeDecisionSettings,
  DecisionError,
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
  signal: AbortSignal;
  redirect: 'error';
};

function fakeFetch(
  responses: Array<{ status: number; body: string; retryAfter?: string }>,
  calls: FetchCall[] = []
): { fetch: DecisionFetch; calls: FetchCall[] } {
  let i = 0;
  const fetch: DecisionFetch = async (url, init) => {
    calls.push({
      url,
      body: JSON.parse(init.body) as Record<string, unknown>,
      auth: init.headers.Authorization,
      signal: init.signal,
      redirect: init.redirect,
    });
    const reply = responses[Math.min(i++, responses.length - 1)];
    return new Response(reply.body, {
      status: reply.status,
      headers:
        reply.retryAfter !== undefined
          ? { 'retry-after': reply.retryAfter }
          : {},
    });
  };
  return { fetch, calls };
}

const measured = JSON.stringify({
  model: 'jev-1.13.0',
  answers: { q: { type: 'noul', noul: 0.9 } },
  usage: { input_tokens: 10, output_tokens: 1 },
});

describe('decision dialect', () => {
  it('maps boolean, choice, and expected-value score questions to System One', () => {
    expect(
      toWireQuestion(booleanQuestion('Is it a test?'), 'systemone')
    ).toEqual({
      type: 'noul',
      instructions: 'Is it a test?',
    });
    expect(
      toWireQuestion(
        booleanQuestion('Is it a test?', 'calls test()'),
        'systemone'
      )
    ).toEqual({
      type: 'noul',
      instructions: 'Is it a test?',
      criteria: { true: 'calls test()' },
    });
    expect(
      toWireQuestion(
        booleanQuestion('Test?', { true: 'yes', false: 'no' }),
        'systemone'
      )
    ).toEqual({
      type: 'noul',
      instructions: 'Test?',
      criteria: { true: 'yes', false: 'no' },
    });
    expect(toWireQuestion(booleanQuestion('Test?', 'yes'), 'port')).toEqual({
      type: 'boolean',
      instructions: 'Test?',
      criteria: { true: 'yes' },
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

  it('keeps measured probabilities and expected-value scores, but not fabricated distributions', () => {
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
      {
        type: 'choice',
        choice: 'hook',
        confidence: null,
        probabilities: null,
      }
    );
    expect(
      readAnswer(
        {
          type: 'score',
          score: 1.7,
          confidence: 0.9,
          probabilities: { '0': 0.1, '1': 0.1, '2': 0.8 },
        },
        'systemone'
      )
    ).toEqual({
      type: 'score',
      score: 1.7,
      confidence: 0.9,
      probabilities: { '0': 0.1, '1': 0.1, '2': 0.8 },
    });
  });

  it('rejects invalid ranges, choices, scores, and probability distributions', () => {
    const choice = choiceQuestion('Which?', { hook: 'yes', util: 'no' });
    const score = scoreQuestion('Score?', ['a', 'b', 'c']);
    expect(readAnswer({ type: 'noul', noul: -0.01 }, 'systemone')).toBeNull();
    expect(readAnswer({ type: 'noul', noul: 1.01 }, 'systemone')).toBeNull();
    expect(
      readAnswer({ type: 'noul', noul: Infinity }, 'systemone')
    ).toBeNull();
    expect(
      readAnswer({ type: 'choice', choice: 'other' }, 'systemone', choice)
    ).toBeNull();
    expect(
      readAnswer(
        { type: 'choice', choice: 'hook', probabilities: { hook: 1.2 } },
        'systemone',
        choice
      )
    ).toBeNull();
    expect(
      readAnswer(
        { type: 'choice', choice: 'hook', probabilities: { other: 0.1 } },
        'systemone',
        choice
      )
    ).toBeNull();
    expect(
      readAnswer({ type: 'score', score: 2.1 }, 'systemone', score)
    ).toBeNull();
    expect(
      readAnswer({ type: 'score', score: NaN }, 'systemone', score)
    ).toBeNull();
    expect(readAnswer({ type: 'choice', choice: 7 }, 'systemone')).toBeNull();
    expect(
      readAnswer(
        { type: 'score', score: 0, probabilities: { '0': 0, '1': 0, '2': 1 } },
        'systemone',
        score
      )
    ).toBeNull();
    expect(
      readAnswer(
        {
          type: 'score',
          score: 1.6,
          probabilities: { '0': 0.1, '1': 0.2, '2': 0.7 },
        },
        'systemone',
        score
      )
    ).toMatchObject({ type: 'score', score: 1.6 });
    const manyProbabilities = Object.fromEntries(
      Array.from({ length: 2000 }, (_, index) => [`k${index}`, 0])
    );
    manyProbabilities.k0 = 1;
    manyProbabilities.k1 = 0.05;
    expect(
      readAnswer(
        { type: 'choice', choice: 'k0', probabilities: manyProbabilities },
        'systemone'
      )
    ).toBeNull();
    expect(
      readAnswer(
        {
          type: 'choice',
          choice: 'hook',
          probabilities: { hook: 0.8, util: 0.8 },
        },
        'systemone',
        choice
      )
    ).toBeNull();
    expect(
      readAnswer(
        { type: 'choice', choice: 'hook', probabilities: { hook: 1 } },
        'systemone',
        choice
      )
    ).toBeNull();
    expect(
      readAnswer(
        { type: 'score', score: 0, probabilities: { '0': 0, '1': 0, '2': 0 } },
        'systemone',
        score
      )
    ).toBeNull();
    expect(
      readAnswer(
        {
          type: 'score',
          score: 1,
          probabilities: { '0': 0.3333, '1': 0.3333, '2': 0.3333 },
        },
        'systemone',
        score
      )
    ).toEqual({
      type: 'score',
      score: 1,
      confidence: null,
      probabilities: { '0': 0.3333, '1': 0.3333, '2': 0.3333 },
    });
  });

  it.each(['port', 'systemone'] as const)(
    'requires measured choices to select a maximum in the %s dialect, allowing ties',
    (dialect) => {
      const question = choiceQuestion('Which?', { a: 'A', b: 'B' });
      expect(
        readAnswer(
          { type: 'choice', choice: 'a', probabilities: { a: 0.1, b: 0.9 } },
          dialect,
          question
        )
      ).toBeNull();
      expect(
        readAnswer(
          { type: 'choice', choice: 'missing', probabilities: { a: 1, b: 0 } },
          dialect
        )
      ).toBeNull();
      for (const choice of ['a', 'b']) {
        expect(
          readAnswer(
            { type: 'choice', choice, probabilities: { a: 0.5, b: 0.5 } },
            dialect,
            question
          )
        ).toMatchObject({ type: 'choice', choice });
      }
      expect(
        readAnswer(
          { type: 'choice', choice: 'b', probabilities: { a: 0.1, b: 0.9 } },
          dialect,
          question
        )
      ).toMatchObject({ type: 'choice', choice: 'b' });
      expect(
        readAnswer({ type: 'choice', choice: 'a' }, dialect, question)
      ).toMatchObject({ type: 'choice', choice: 'a', probabilities: null });
    }
  );

  it.each([
    { invalid: 1 },
    { NaN: 1 },
    { '01': 0, '0': 1 },
    { '-1': 0, '0': 1 },
    { '1.5': 0, '0': 1 },
    { '1e2': 0, '0': 1 },
    { Infinity: 0, '0': 1 },
    { '0': 1, '2': 0 },
    { '0': 1, ['9'.repeat(400)]: 0 },
  ])(
    'rejects malformed score levels %j without an expected question',
    (probabilities) => {
      const answer = { type: 'score', score: 0, probabilities };
      for (const dialect of ['port', 'systemone'] as const) {
        expect(readAnswer(answer, dialect)).toBeNull();
        expect(() =>
          parseEnvelope(
            JSON.stringify({ answers: { q: answer } }),
            'test',
            (raw) => readAnswer(raw, dialect)
          )
        ).toThrow(DecisionError);
      }
    }
  );

  it('keeps valid measured and unmeasured scores without an expected question', () => {
    for (const dialect of ['port', 'systemone'] as const) {
      expect(
        readAnswer(
          {
            type: 'score',
            score: 1.7,
            probabilities: { '0': 0.1, '1': 0.1, '2': 0.8 },
          },
          dialect
        )
      ).toMatchObject({ type: 'score', score: 1.7 });
      expect(readAnswer({ type: 'score', score: 99 }, dialect)).toMatchObject({
        type: 'score',
        score: 99,
        probabilities: null,
      });
    }
  });

  it('rejects malformed envelopes without leaking the body', () => {
    expect(() =>
      parseEnvelope('not json', 'jev', (a) => readAnswer(a, 'systemone'))
    ).toThrow(DecisionError);
    expect(() => parseEnvelope('{"model":"m"}', 'jev', () => null)).toThrow(
      /no answers/
    );
    expect(() =>
      parseEnvelope('{"answers":{"q":{"type":"noul","noul":2}}}', 'jev', (a) =>
        readAnswer(a, 'systemone')
      )
    ).toThrow(/invalid answer/);
  });
});

describe('HTTP decisionModel', () => {
  it('posts Jev-shaped questions and keeps only measured answer probabilities', async () => {
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
            severity: {
              type: 'score',
              score: 1.6,
              probabilities: { '0': 0.1, '1': 0.2, '2': 0.7 },
            },
          },
          usage: { input_tokens: 10, output_tokens: 1 },
        }),
      },
    ]);
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example/v1/systemone',
      apiKey: 'k',
      model: 'jev-latest',
      dialect: 'systemone',
      fetch,
    });
    const result = await decisionModel.decide({
      state: { path: 'x.ts' },
      questions: {
        role: choiceQuestion('Which?', { hook: 'a hook', util: 'a util' }),
        is_test: booleanQuestion('test?'),
        severity: scoreQuestion('Severity?', ['low', 'medium', 'high']),
      },
    });
    expect(calls).toHaveLength(1);
    expect(calls[0].auth).toBe('Bearer k');
    expect(calls[0].redirect).toBe('error');
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
        severity: {
          type: 'score',
          instructions: 'Severity?',
          criteria: ['low', 'medium', 'high'],
        },
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
    expect(result.answers.severity).toEqual({
      type: 'score',
      score: 1.6,
      confidence: null,
      probabilities: { '0': 0.1, '1': 0.2, '2': 0.7 },
    });
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 1 });
  });

  it('validates against the question sent, not a caller mutation during fetch', async () => {
    let reply: ((response: Response) => void) | undefined;
    let requested: (() => void) | undefined;
    const sent = new Promise<void>((resolve) => {
      requested = resolve;
    });
    const pending = new Promise<Response>((resolve) => {
      reply = resolve;
    });
    let payload = '';
    const fetch: DecisionFetch = async (_url, init) => {
      payload = init.body;
      requested?.();
      return pending;
    };
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example/v1/systemone',
      apiKey: 'k',
      dialect: 'systemone',
      fetch,
    });
    const pick = choiceQuestion('Pick?', { first: 'First', second: 'Second' });
    const questions = { pick };
    const resultPromise = decisionModel.decide({ state: 'hello', questions });
    await sent;
    Object.assign(pick, { type: 'boolean' });
    delete pick.criteria.first;
    pick.criteria.third = 'Third';
    reply?.(
      new Response(
        JSON.stringify({
          answers: {
            pick: {
              type: 'choice',
              choice: 'first',
              probabilities: { first: 0.8, second: 0.2 },
            },
          },
        })
      )
    );
    expect((await resultPromise).answers.pick).toMatchObject({
      type: 'choice',
      choice: 'first',
    });
    expect(JSON.parse(payload)).toMatchObject({
      questions: {
        pick: {
          type: 'choice',
          criteria: { first: 'First', second: 'Second' },
        },
      },
    });
  });

  it('sends the same wire format to unauthenticated Laya without pinning a model', async () => {
    const { fetch, calls } = fakeFetch([{ status: 200, body: measured }]);
    const settings = mergeDecisionSettings(decisionPreset('laya'), {
      baseURL: 'http://localhost:8000/v1/systemone',
    });
    const decisionModel = createDecisionModel(settings, undefined, {
      providerId: 'laya',
      fetch,
    });
    const result = await decisionModel.decide({
      state: 'bill me twice',
      questions: { q: booleanQuestion('Was billing mentioned?') },
    });
    expect(calls[0]).toMatchObject({
      url: 'http://localhost:8000/v1/systemone',
      auth: undefined,
      body: {
        state: 'bill me twice',
        questions: {
          q: { type: 'noul', instructions: 'Was billing mentioned?' },
        },
      },
    });
    expect(calls[0].body).not.toHaveProperty('model');
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(decisionPreset('laya')).toMatchObject({
      dialect: 'systemone',
      requiresAuth: false,
    });
    expect(decisionPreset('laya')).not.toHaveProperty('baseURL');
    expect(decisionPreset('laya')).not.toHaveProperty('model');
  });

  it('supports optional Laya bearer auth and keeps mandatory Jev credentials', async () => {
    const { fetch, calls } = fakeFetch([{ status: 200, body: measured }]);
    const laya = createDecisionModel(
      mergeDecisionSettings(decisionPreset('laya'), {
        baseURL: 'https://local.example/v1/systemone',
      }),
      'laya-secret',
      { fetch }
    );
    await laya.decide({ state: {}, questions: { q: booleanQuestion('?') } });
    expect(calls[0].auth).toBe('Bearer laya-secret');
    expect(() =>
      createDecisionModel(decisionPreset('typesafe') ?? {}, undefined)
    ).toThrow(/API key/);
    const required = mergeDecisionSettings(decisionPreset('typesafe'), {
      requiresAuth: false,
    });
    expect(required.requiresAuth).toBe(true);
    expect(() => createDecisionModel(required, undefined)).toThrow(/API key/);
    expect(() =>
      createDecisionModel(
        mergeDecisionSettings(decisionPreset('laya'), {
          baseURL: 'https://local.example/v1/systemone',
          requiresAuth: true,
        }),
        undefined
      )
    ).toThrow(/API key/);
    expect(() =>
      createDecisionModel(decisionPreset('laya') ?? {}, undefined)
    ).toThrow(/endpoint/);
    expect(() =>
      createHttpDecisionModel({ endpoint: 'https://s1.example', apiKey: '' })
    ).toThrow(/API key/);
    expect(decisionPreset('typesafe')).toMatchObject({
      baseURL: 'https://api.typesafe.ai/v1/systemone',
      model: 'jev-latest',
      dialect: 'systemone',
    });
    expect(decisionPreset('nope')).toBeNull();
    expect(decisionPreset('__proto__')).toBeNull();
    expect(decisionPreset('constructor')).toBeNull();
  });

  it('does not let one tenant mutate shared preset endpoints or authentication', () => {
    expect(Object.isFrozen(DECISION_PRESETS)).toBe(true);
    expect(Object.isFrozen(DECISION_PRESETS.typesafe)).toBe(true);
    const first = decisionPreset('typesafe');
    const second = decisionPreset('typesafe');
    expect(first).not.toBe(second);
    if (first === null) {
      throw new Error('TypeSafe preset disappeared');
    }
    first.baseURL = 'https://untrusted.example/v1/systemone';
    first.requiresAuth = false;
    expect(decisionPreset('typesafe')).toMatchObject({
      baseURL: 'https://api.typesafe.ai/v1/systemone',
      requiresAuth: true,
    });
    expect(() =>
      createDecisionModel(decisionPreset('typesafe') ?? {}, undefined)
    ).toThrow(/API key/);
  });

  it('nests the Cloudflare body and returns unknown usage rather than zero tokens', async () => {
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
    const decisionModel = createDecisionModel(
      mergeDecisionSettings(decisionPreset('cloudflare'), {
        baseURL: 'https://cf.example/run',
      }),
      'k',
      { fetch }
    );
    const result = await decisionModel.decide({
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
    expect(result.usage).toBeNull();
  });

  it('marks missing answers as undefined and rejects unknown, wrong-type, and invalid answers', async () => {
    const questions = {
      q: booleanQuestion('?'),
      other: booleanQuestion('Another?'),
    };
    const partial = fakeFetch([{ status: 200, body: measured }]);
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      dialect: 'systemone',
      fetch: partial.fetch,
    });
    const result = await decisionModel.decide({ state: {}, questions });
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(result.answers.other).toBeUndefined();

    for (const answers of [
      {},
      { extra: { type: 'noul', noul: 0.5 } },
      { q: { type: 'choice', choice: 'x' } },
      { q: { type: 'noul', noul: -0.1 } },
    ]) {
      const invalid = fakeFetch([
        { status: 200, body: JSON.stringify({ answers }) },
      ]);
      await expect(
        createHttpDecisionModel({
          endpoint: 'https://s1.example',
          apiKey: 'k',
          dialect: 'systemone',
          fetch: invalid.fetch,
        }).decide({ state: {}, questions: { q: booleanQuestion('?') } })
      ).rejects.toMatchObject({ failure: 'malformed_response' });
    }
  });

  it('rejects a contradictory measured choice without firing the answer hook', async () => {
    const onAnswered = jest.fn();
    const { fetch, calls } = fakeFetch([
      {
        status: 200,
        body: JSON.stringify({
          answers: {
            q: {
              type: 'choice',
              choice: 'a',
              probabilities: { a: 0.1, b: 0.9 },
            },
          },
        }),
      },
    ]);
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      dialect: 'systemone',
      fetch,
      onAnswered,
    });
    await expect(
      decisionModel.decide({
        state: {},
        questions: { q: choiceQuestion('Which?', { a: 'A', b: 'B' }) },
      })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
    expect(calls).toHaveLength(1);
    expect(onAnswered).not.toHaveBeenCalled();
  });

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
    'rejects malformed boolean criteria %j before HTTP transport',
    async (criteria) => {
      const { fetch, calls } = fakeFetch([{ status: 200, body: measured }]);
      const apiKey = jest.fn(async () => 'k');
      const decisionModel = createHttpDecisionModel({
        endpoint: 'https://s1.example',
        apiKey,
        dialect: 'systemone',
        fetch,
      });
      const question = JSON.parse(
        JSON.stringify({ type: 'boolean', instructions: '?', criteria })
      ) as DecisionQuestion;
      await expect(
        decisionModel.decide({ state: {}, questions: { q: question } })
      ).rejects.toMatchObject({ failure: 'bad_request' });
      expect(calls).toHaveLength(0);
      expect(apiKey).not.toHaveBeenCalled();
    }
  );

  it.each([
    undefined,
    '',
    'Matches',
    {},
    { true: 'Matches' },
    { false: 'Does not match' },
    { true: 'Matches', false: 'Does not match' },
    { true: ['Matches'], false: { text: 'Does not match' } },
  ])('accepts supported boolean criteria %j', async (criteria) => {
    const { fetch, calls } = fakeFetch([{ status: 200, body: measured }]);
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      dialect: 'systemone',
      fetch,
    });
    const result = await decisionModel.decide({
      state: {},
      questions: { q: booleanQuestion('?', criteria) },
    });
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(calls).toHaveLength(1);
  });

  it('rejects malformed question IDs before sending a request', async () => {
    const { fetch, calls } = fakeFetch([{ status: 200, body: measured }]);
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      fetch,
    });
    await expect(
      decisionModel.decide({
        state: {},
        questions: { ['bad\nid']: booleanQuestion('Is this valid?') },
      })
    ).rejects.toMatchObject({ failure: 'bad_request' });
    expect(calls).toHaveLength(0);
  });

  it('retries a 429 but never retries a 403, and refreshes a minted key once for a 401', async () => {
    const forbidden = fakeFetch([
      { status: 403, body: 'private echoed content and key' },
    ]);
    await expect(
      createHttpDecisionModel({
        endpoint: 'https://s1.example',
        apiKey: 'secret',
        fetch: forbidden.fetch,
      }).decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'unauthorized', status: 403 });
    expect(forbidden.calls).toHaveLength(1);

    const limited = fakeFetch([
      { status: 429, body: 'slow down', retryAfter: '0' },
      {
        status: 200,
        body: JSON.stringify({
          answers: { q: { type: 'boolean', probability: 0.4 } },
        }),
      },
    ]);
    const patient = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      fetch: limited.fetch,
      sleep: async () => {},
    });
    expect(
      (
        await patient.decide({
          state: {},
          questions: { q: booleanQuestion('?') },
        })
      ).answers.q
    ).toEqual({
      type: 'boolean',
      probability: 0.4,
    });
    expect(limited.calls).toHaveLength(2);

    const seen: boolean[] = [];
    const expiring = fakeFetch([
      { status: 401, body: 'expired' },
      { status: 200, body: measured },
    ]);
    const credential = async ({ refresh }: { refresh: boolean }) => {
      seen.push(refresh);
      return refresh ? 'new' : 'old';
    };
    await createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: credential,
      dialect: 'systemone',
      fetch: expiring.fetch,
      maxRetries: 0,
    }).decide({ state: {}, questions: { q: booleanQuestion('?') } });
    expect(expiring.calls.map((call) => call.auth)).toEqual([
      'Bearer old',
      'Bearer new',
    ]);
    expect(seen).toEqual([false, true]);
  });

  it('mints one token per call and retains a refreshed token across subsequent retries', async () => {
    const auths: Array<string | undefined> = [];
    const minted: boolean[] = [];
    const statuses = [401, 503, 429, 200];
    const fetch: DecisionFetch = async (_url, init) => {
      auths.push(init.headers.Authorization);
      const status = statuses[auths.length - 1];
      return new Response(status === 200 ? measured : 'retriable', { status });
    };
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: async ({ refresh }) => {
        minted.push(refresh);
        return refresh ? 'fresh' : 'stale';
      },
      dialect: 'systemone',
      maxRetries: 2,
      fetch,
      sleep: async () => {},
    });
    const result = await decisionModel.decide({
      state: {},
      questions: { q: booleanQuestion('?') },
    });
    expect(result.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(auths).toEqual([
      'Bearer stale',
      'Bearer fresh',
      'Bearer fresh',
      'Bearer fresh',
    ]);
    expect(minted).toEqual([false, true]);

    const retries = fakeFetch([
      { status: 429, body: 'slow down', retryAfter: '0' },
      { status: 200, body: measured },
    ]);
    const plainMints: boolean[] = [];
    const plain = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: async ({ refresh }) => {
        plainMints.push(refresh);
        return 'one-token';
      },
      dialect: 'systemone',
      fetch: retries.fetch,
      sleep: async () => {},
    });
    await plain.decide({ state: {}, questions: { q: booleanQuestion('?') } });
    expect(plainMints).toEqual([false]);
    expect(retries.calls.map(({ auth }) => auth)).toEqual([
      'Bearer one-token',
      'Bearer one-token',
    ]);
  });

  it('bounds hanging credential minting, caller abort, and non-cooperative response reading', async () => {
    const calls: FetchCall[] = [];
    const waiting = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: async () => new Promise<string>(() => {}),
      timeoutMs: 25,
      fetch: fakeFetch([{ status: 200, body: measured }], calls).fetch,
    });
    await expect(
      waiting.decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'timeout' });
    expect(calls).toHaveLength(0);

    const cancelled = new AbortController();
    cancelled.abort();
    const unvisited = {
      get secret(): string {
        throw new Error('state was serialized');
      },
    };
    await expect(
      waiting.decide({
        state: unvisited,
        questions: { q: booleanQuestion('?') },
        signal: cancelled.signal,
      })
    ).rejects.toMatchObject({ failure: 'aborted' });
    const duringMint = new AbortController();
    const pending = waiting.decide({
      state: {},
      questions: { q: booleanQuestion('?') },
      signal: duringMint.signal,
    });
    duringMint.abort();
    await expect(pending).rejects.toMatchObject({ failure: 'aborted' });

    const hangingFetch: DecisionFetch = async () =>
      new Promise<Response>(() => {});
    await expect(
      createHttpDecisionModel({
        endpoint: 'https://s1.example',
        apiKey: 'k',
        timeoutMs: 25,
        fetch: hangingFetch,
      }).decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'timeout' });

    const hangingRead: DecisionFetch = async () =>
      new Response(new ReadableStream<Uint8Array>({ start() {} }), {
        status: 200,
      });
    const reading = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      timeoutMs: 25,
      fetch: hangingRead,
    });
    await expect(
      reading.decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'timeout' });
  });

  it('aborts the fetch signal when the monotonic deadline expires before the timer runs', async () => {
    let seenSignal: AbortSignal | undefined;
    const fetch: DecisionFetch = async (_url, init) => {
      seenSignal = init.signal;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 15);
      return new Response(measured, { status: 200 });
    };
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      timeoutMs: 2,
      fetch,
    });
    await expect(
      decisionModel.decide({
        state: {},
        questions: { q: booleanQuestion('?') },
      })
    ).rejects.toMatchObject({ failure: 'timeout' });
    expect(seenSignal?.aborted).toBe(true);
  });

  it('bounds synchronous request preparation even when a state getter throws', async () => {
    const fetch: DecisionFetch = jest.fn(
      async () => new Response(measured, { status: 200 })
    );
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      timeoutMs: 2,
      fetch,
    });
    const state = {
      get body(): string {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 15);
        throw new Error('private request content');
      },
    };
    await expect(
      decisionModel.decide({ state, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'timeout' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('bounds backoff and response size, sanitizes failures, and hooks only parsed answers', async () => {
    const onAnswered = jest.fn();
    const limited = fakeFetch([
      { status: 429, body: 'echo secret', retryAfter: '0' },
    ]);
    const backedOff = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'secret',
      timeoutMs: 25,
      fetch: limited.fetch,
      sleep: async () => new Promise<void>(() => {}),
      onAnswered,
    });
    await expect(
      backedOff.decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'timeout' });
    expect(limited.calls).toHaveLength(1);
    expect(onAnswered).not.toHaveBeenCalled();

    const oversized = fakeFetch([
      { status: 200, body: 'x'.repeat(256 * 1024 + 1) },
    ]);
    await expect(
      createHttpDecisionModel({
        endpoint: 'https://s1.example',
        apiKey: 'secret',
        fetch: oversized.fetch,
        onAnswered,
      }).decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
    expect(onAnswered).not.toHaveBeenCalled();

    const failed = fakeFetch([
      { status: 500, body: 'secret and echoed user content' },
    ]);
    try {
      await createHttpDecisionModel({
        endpoint: 'https://s1.example',
        apiKey: 'secret',
        fetch: failed.fetch,
        maxRetries: 0,
        onAnswered,
      }).decide({ state: {}, questions: { q: booleanQuestion('?') } });
      throw new Error('expected an HTTP failure');
    } catch (error) {
      expect(error).toMatchObject({ failure: 'server_error' });
      expect(String(error)).not.toMatch(/secret|echoed/);
    }
    const invalid = fakeFetch([
      { status: 200, body: 'private echoed user content' },
    ]);
    await expect(
      createHttpDecisionModel({
        endpoint: 'https://s1.example',
        apiKey: 'secret',
        fetch: invalid.fetch,
        onAnswered,
      }).decide({ state: {}, questions: { q: booleanQuestion('?') } })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
    expect(onAnswered).not.toHaveBeenCalled();

    const ok = fakeFetch([{ status: 200, body: measured }]);
    await createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: 'k',
      dialect: 'systemone',
      fetch: ok.fetch,
      onAnswered,
    }).decide({
      state: {},
      questions: { q: booleanQuestion('?') },
      label: 'gate',
    });
    expect(onAnswered).toHaveBeenCalledTimes(1);
    expect(onAnswered).toHaveBeenCalledWith('gate', expect.any(Number));
  });

  it('keeps concurrent credential refreshes and request signals independent', async () => {
    const seen: Array<{ refresh: boolean; signal?: AbortSignal }> = [];
    const calls: FetchCall[] = [];
    const credential = async (options: {
      refresh: boolean;
      signal?: AbortSignal;
    }) => {
      seen.push(options);
      return options.refresh ? 'new' : 'old';
    };
    const fetch: DecisionFetch = async (url, init) => {
      calls.push({
        url,
        body: JSON.parse(init.body) as Record<string, unknown>,
        auth: init.headers.Authorization,
        signal: init.signal,
        redirect: init.redirect,
      });
      await Promise.resolve();
      return new Response(
        init.headers.Authorization === 'Bearer old' ? 'expired' : measured,
        {
          status: init.headers.Authorization === 'Bearer old' ? 401 : 200,
        }
      );
    };
    const decisionModel = createHttpDecisionModel({
      endpoint: 'https://s1.example',
      apiKey: credential,
      dialect: 'systemone',
      fetch,
      maxRetries: 0,
    });
    const [first, second] = await Promise.all([
      decisionModel.decide({
        state: 'one',
        questions: { q: booleanQuestion('?') },
      }),
      decisionModel.decide({
        state: 'two',
        questions: { q: booleanQuestion('?') },
      }),
    ]);
    expect(first.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(second.answers.q).toEqual({ type: 'boolean', probability: 0.9 });
    expect(seen.filter((entry) => entry.refresh)).toHaveLength(2);
    expect(seen.filter((entry) => !entry.refresh)).toHaveLength(2);
    expect(new Set(seen.map((entry) => entry.signal)).size).toBe(2);
    expect(new Set(calls.map((call) => call.signal)).size).toBe(2);
    expect(calls).toHaveLength(4);
  });
});

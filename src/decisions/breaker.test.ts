import type { DecisionFetch, DecisionModel, DecisionCredential } from './core';
import {
  probe,
  redact,
  withBreaker,
  createBreaker,
  DecisionError,
  booleanQuestion,
  createHttpDecisionModel,
} from './core';

const ENDPOINT = 'https://decisions.example/v1/systemone';
const KEY = 'sk-test-0123456789abcdefghijklmnopqrstuvwxyz';

const answered = JSON.stringify({
  model: 'jev-1.13.0',
  answers: { q: { type: 'noul', noul: 0.9 } },
});

function replies(statuses: number[]): {
  fetch: DecisionFetch;
  auth: Array<string | undefined>;
} {
  const auth: Array<string | undefined> = [];
  let index = 0;
  const fetch: DecisionFetch = async (_url, init) => {
    auth.push(init.headers.Authorization);
    const status = statuses[Math.min(index++, statuses.length - 1)];
    return new Response(status === 200 ? answered : 'refused', { status });
  };
  return { fetch, auth };
}

function fused(
  statuses: number[],
  apiKey: DecisionCredential,
  maxRetries = 2
): {
  model: ReturnType<typeof withBreaker>;
  auth: Array<string | undefined>;
  sleeps: number[];
} {
  const { fetch, auth } = replies(statuses);
  const sleeps: number[] = [];
  const model = withBreaker(
    createHttpDecisionModel({
      endpoint: ENDPOINT,
      apiKey,
      dialect: 'systemone',
      model: 'jev-1.13.0',
      maxRetries,
      fetch,
      sleep: async (ms) => {
        sleeps.push(ms);
      },
    })
  );
  return { model, auth, sleeps };
}

const request = {
  state: 'synthetic state',
  questions: { q: booleanQuestion('Is this synthetic?') },
};

async function failure(model: DecisionModel): Promise<DecisionError> {
  const error = await model.decide(request).then(
    () => undefined,
    (thrown: unknown) => thrown
  );
  if (!(error instanceof DecisionError)) {
    throw new Error('expected a DecisionError');
  }
  return error;
}

describe('withBreaker', () => {
  it('opens on a 401 with a string key and refuses later calls without a request', async () => {
    const { model, auth } = fused([401, 200], KEY);

    const first = await failure(model);
    expect(first).toMatchObject({
      failure: 'unauthorized',
      status: 401,
      circuitOpen: false,
    });
    expect(auth).toHaveLength(1);
    expect(model.breaker.open).toBe(true);
    expect(model.breaker.error).toBe(first);

    const second = await failure(model);
    const third = await failure(model);
    expect(auth).toHaveLength(1);
    for (const refused of [second, third]) {
      expect(refused).toMatchObject({
        failure: 'unauthorized',
        status: 401,
        circuitOpen: true,
        provider: 'http',
      });
      expect(refused.message).not.toContain(KEY);
    }
  });

  it('refreshes a credential function once, retries, then opens on the second 401', async () => {
    const minted: boolean[] = [];
    const credential: DecisionCredential = async ({ refresh }) => {
      minted.push(refresh);
      return `token-${minted.length}`;
    };
    const { model, auth } = fused([401, 401, 200], credential);

    const first = await failure(model);
    expect(first).toMatchObject({ failure: 'unauthorized', status: 401 });
    expect(minted).toEqual([false, true]);
    expect(auth).toEqual(['Bearer token-1', 'Bearer token-2']);
    expect(model.breaker.open).toBe(true);

    const refused = await failure(model);
    expect(refused.circuitOpen).toBe(true);
    expect(minted).toEqual([false, true]);
    expect(auth).toHaveLength(2);
  });

  it('opens on a 402 quota failure after exactly one request', async () => {
    const { model, auth, sleeps } = fused([402, 200], KEY);

    const first = await failure(model);
    expect(first).toMatchObject({ failure: 'quota', status: 402 });
    expect(auth).toHaveLength(1);
    expect(sleeps).toEqual([]);
    expect(model.breaker.open).toBe(true);

    const refused = await failure(model);
    expect(refused).toMatchObject({
      failure: 'quota',
      status: 402,
      circuitOpen: true,
    });
    expect(auth).toHaveLength(1);
  });

  it.each([
    [429, 'rate_limited'],
    [500, 'server_error'],
  ])(
    'retries a %d exactly maxRetries times and stays closed',
    async (status, expected) => {
      const { model, auth, sleeps } = fused([status], KEY, 3);

      const first = await failure(model);
      expect(first).toMatchObject({ failure: expected, status });
      expect(auth).toHaveLength(4);
      expect(sleeps).toHaveLength(3);
      expect(model.breaker.open).toBe(false);

      await failure(model);
      expect(auth).toHaveLength(8);
      expect(model.breaker.open).toBe(false);
    }
  );

  it('sends requests again after an explicit reset', async () => {
    const { model, auth } = fused([401, 200], KEY);
    await failure(model);
    expect(model.breaker.open).toBe(true);

    model.breaker.reset();
    expect(model.breaker.open).toBe(false);
    expect(model.breaker.error).toBeNull();
    const result = await model.decide(request);
    expect(result.model).toBe('jev-1.13.0');
    expect(auth).toHaveLength(2);
  });

  it('shares one breaker between models that share a credential', async () => {
    const breaker = createBreaker();
    const refused = replies([401]);
    const healthy = replies([200]);
    const options = {
      endpoint: ENDPOINT,
      apiKey: KEY,
      dialect: 'systemone' as const,
    };
    const first = withBreaker(
      createHttpDecisionModel({ ...options, fetch: refused.fetch }),
      breaker
    );
    const second = withBreaker(
      createHttpDecisionModel({ ...options, fetch: healthy.fetch }),
      breaker
    );

    await failure(first);
    const error = await failure(second);
    expect(error.circuitOpen).toBe(true);
    expect(healthy.auth).toHaveLength(0);
  });

  it('keeps the first error that tripped it', () => {
    const breaker = createBreaker();
    const first = new DecisionError('quota', 'first', { status: 402 });
    breaker.trip(first);
    breaker.trip(new DecisionError('unauthorized', 'second', { status: 401 }));
    expect(breaker.error).toBe(first);
  });
});

describe('probe', () => {
  it('asks one tiny boolean question and reports the answering model', async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const fetch: DecisionFetch = async (_url, init) => {
      bodies.push(JSON.parse(init.body) as Record<string, unknown>);
      return new Response(
        JSON.stringify({
          model: 'jev-1.13.0',
          answers: { ok: { type: 'noul', noul: 0.97 } },
        }),
        { status: 200 }
      );
    };
    const model = createHttpDecisionModel({
      endpoint: ENDPOINT,
      apiKey: KEY,
      dialect: 'systemone',
      model: 'jev-latest',
      fetch,
    });

    await expect(probe(model)).resolves.toEqual({
      ok: true,
      model: 'jev-1.13.0',
    });
    expect(bodies).toHaveLength(1);
    expect(bodies[0]).toEqual({
      model: 'jev-latest',
      state: 'decision model readiness probe',
      questions: {
        ok: { type: 'noul', instructions: 'Is this text a readiness probe?' },
      },
    });
  });

  it('returns a typed failure instead of throwing, and trips a breaker', async () => {
    const { model, auth } = fused([402], KEY);

    await expect(probe(model)).resolves.toEqual({
      ok: false,
      failure: 'quota',
      status: 402,
      circuitOpen: false,
      detail: 'decision model returned HTTP 402',
    });
    await expect(probe(model)).resolves.toMatchObject({
      ok: false,
      failure: 'quota',
      circuitOpen: true,
    });
    expect(auth).toHaveLength(1);
  });

  it('reads a non-SDK exception as a network failure without its message', async () => {
    const broken: DecisionModel = {
      id: 'broken',
      model: 'broken',
      decide: async () => {
        throw new Error(`socket closed for Bearer ${KEY}`);
      },
    };
    const result = await probe(broken);
    expect(result).toEqual({
      ok: false,
      failure: 'network',
      circuitOpen: false,
      detail: 'decision model probe failed',
    });
  });

  it('redacts listed secrets from the failure detail', async () => {
    const leaky: DecisionModel = {
      id: 'leaky',
      model: 'leaky',
      decide: async () => {
        throw new DecisionError('unauthorized', 'token short-1 was refused');
      },
    };
    const result = await probe(leaky, { secrets: ['short-1'] });
    expect(result).toMatchObject({
      ok: false,
      detail: 'token [REDACTED] was refused',
    });
  });
});

describe('redact', () => {
  it('removes listed secrets, bearer values, JWTs, and long opaque tokens', () => {
    const jwt =
      'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl';
    const text = [
      'Authorization: Bearer abc.def',
      `jwt=${jwt}`,
      `key=${KEY}`,
      'pin=hunter22',
    ].join(' ');
    const result = redact(text, ['hunter22', '', null, undefined]);
    expect(result).toBe(
      'Authorization: Bearer [REDACTED] jwt=[REDACTED] key=[REDACTED] pin=[REDACTED]'
    );
    for (const secret of ['abc.def', jwt, KEY, 'hunter22']) {
      expect(result).not.toContain(secret);
    }
  });

  it('replaces a longer secret before a shorter one it contains', () => {
    expect(redact('a=secret-value-long', ['secret', 'secret-value-long'])).toBe(
      'a=[REDACTED]'
    );
  });
});

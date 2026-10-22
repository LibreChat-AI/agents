import http from 'node:http';
import type { AddressInfo } from 'node:net';
import type {
  DecisionModel,
  DecisionRequest,
  DecisionCredential,
  DecisionProviderSettings,
} from '@/decisions';
import type { RankRequest, RerankOutcome, OutcomeReranker } from './types';
import type { FallbackStep } from './core';
import {
  probePin,
  rerankCost,
  noneReranker,
  withFallback,
  estimatedUsage,
  createScoreCache,
  toOutcomeReranker,
  corpusStats,
  createBm25Reranker,
  createSystemOneReranker,
  createCrossEncoderReranker,
} from './core';
import {
  withBreaker,
  createBreaker,
  DecisionError,
  decisionPreset,
  createDecisionModel,
  createHttpDecisionModel,
} from '@/decisions';

type Json = { [key: string]: unknown };

interface Call {
  path: string;
  body: Json;
  authorization?: string;
}

type Reply =
  | 'hang'
  | {
      status?: number;
      body?: unknown;
      raw?: string;
      delayMs?: number;
      headers?: Record<string, string>;
      /** Sends the headers and part of the body, then drops the connection. */
      truncate?: boolean;
    };

interface Stub {
  url: string;
  calls: Call[];
  maxConcurrent: number;
  close(): Promise<void>;
}

const servers: Stub[] = [];

/** `task`'s result, or `pending` when it has not settled after `ms`; the timer never outlives the race. */
async function within<T>(ms: number, task: Promise<T>): Promise<T | 'pending'> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      task,
      new Promise<'pending'>((resolve) => {
        timer = setTimeout(() => resolve('pending'), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

/** A real HTTP server: requests leave the process's fetch and come back over a socket. */
async function stub(reply: (call: Call, nth: number) => Reply): Promise<Stub> {
  const calls: Call[] = [];
  let active = 0;
  const state = { maxConcurrent: 0 };
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const call: Call = {
        path: req.url ?? '',
        body: JSON.parse(Buffer.concat(chunks).toString('utf8')) as Json,
        authorization: req.headers.authorization,
      };
      calls.push(call);
      active += 1;
      state.maxConcurrent = Math.max(state.maxConcurrent, active);
      res.on('close', () => {
        active -= 1;
      });
      const answer = reply(call, calls.length - 1);
      if (answer === 'hang') {
        return;
      }
      setTimeout(() => {
        if (res.destroyed) {
          return;
        }
        if (answer.truncate === true) {
          res.writeHead(answer.status ?? 200, {
            'content-type': 'application/json',
            'content-length': '1000',
          });
          res.write('{"model":');
          setTimeout(() => res.destroy(), 5);
          return;
        }
        res.writeHead(answer.status ?? 200, {
          'content-type': 'application/json',
          ...answer.headers,
        });
        res.end(answer.raw ?? JSON.stringify(answer.body ?? {}));
      }, answer.delayMs ?? 0);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  const handle: Stub = {
    url: `http://127.0.0.1:${port}`,
    calls,
    get maxConcurrent(): number {
      return state.maxConcurrent;
    },
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
  servers.push(handle);
  return handle;
}

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.close()));
});

const rubric = [
  'Unrelated to the query',
  'Relevant context but does not answer the query',
  'Directly answers the query',
  'The answer itself',
];

function jev(
  url: string,
  settings: DecisionProviderSettings = {},
  apiKey?: DecisionCredential
): DecisionModel {
  return createDecisionModel(
    {
      baseURL: `${url}/v1/systemone`,
      model: 'jev-1.13.0',
      dialect: 'systemone',
      requiresAuth: apiKey !== undefined,
      maxRetries: 0,
      ...settings,
    },
    apiKey,
    { providerId: 'jev' }
  );
}

const documentOf = (call: Call): string =>
  ((call.body.state as Json).document as string | undefined) ?? '';

/** Answers one System One score question per request, echoing the requested model. */
function scored(
  scores: Record<string, number>,
  echo?: string
): (call: Call) => Reply {
  return (call) => ({
    body: {
      model: echo ?? call.body.model,
      answers: {
        relevance: { type: 'score', score: scores[documentOf(call)] },
      },
      usage: { input_tokens: 10, output_tokens: 1 },
    },
  });
}

const order = (outcome: RerankOutcome): number[] =>
  outcome.results.map((entry) => entry.index);

const emptyUsage = {
  calls: 0,
  inputTokens: 0,
  outputTokens: 0,
  searchUnits: 0,
  cached: 0,
  estimatedTokens: 0,
  estimatedSearchUnits: 0,
};

/** Blocks the thread, as a slow synchronous host callback does. */
function spin(ms: number): void {
  const end = performance.now() + ms;
  while (performance.now() < end) {
    // Busy-wait: no timer can fire meanwhile.
  }
}

/** Rejections nobody handled while `run` and the turns after it ran. */
async function unhandledDuring(
  run: () => Promise<unknown>
): Promise<unknown[]> {
  const seen: unknown[] = [];
  const listener = (reason: unknown): void => {
    seen.push(reason);
  };
  process.on('unhandledRejection', listener);
  try {
    await run();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    process.off('unhandledRejection', listener);
  }
  return seen;
}

/** An in-process decision model answering `score(request)` after `delayMs`, recording when each request was sent. */
function fakeJev(
  score: (request: DecisionRequest) => number,
  delayMs = 5
): { model: DecisionModel; sent: DecisionRequest[] } {
  const sent: DecisionRequest[] = [];
  return {
    sent,
    model: {
      id: 'fake-jev',
      model: 'jev-1.13.0',
      decide: (request) => {
        sent.push(request);
        return new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                model: 'jev-1.13.0',
                answers: {
                  relevance: {
                    type: 'score',
                    score: score(request),
                    confidence: null,
                    probabilities: null,
                  },
                },
                usage: { inputTokens: 10, outputTokens: 1 },
              }),
            delayMs
          )
        );
      },
    },
  };
}

const SCORES = { alpha: 0.4, beta: 2.6, gamma: 1.2, delta: 2.9 };
const DOCUMENTS = ['alpha', 'beta', 'gamma', 'delta'];

describe('System One per-document ranking', () => {
  it('is the default: one request and one score question per document, so the ranking does not follow input order', async () => {
    const server = await stub(scored(SCORES));
    const reranker = createSystemOneReranker({
      model: jev(server.url),
      rubric,
    });
    expect(reranker.mode).toBe('document');
    const forward = await reranker.rank({ query: 'q', documents: DOCUMENTS });
    expect(server.calls).toHaveLength(4);
    for (const call of server.calls) {
      expect(call.path).toBe('/v1/systemone');
      expect(call.body).toMatchObject({
        model: 'jev-1.13.0',
        state: { query: 'q' },
        questions: { relevance: { type: 'score', criteria: rubric } },
      });
      expect(Object.keys(call.body.questions as Json)).toEqual(['relevance']);
    }
    expect(server.calls.map(documentOf).sort()).toEqual([...DOCUMENTS].sort());
    expect(forward).toMatchObject({
      status: 'ok',
      reason: null,
      provider: 'systemone',
      model: 'jev-1.13.0',
      usage: {
        calls: 4,
        inputTokens: 40,
        outputTokens: 4,
        estimatedTokens: 0,
      },
    });
    expect(forward.results).toEqual([
      { index: 3, score: 2.9 },
      { index: 1, score: 2.6 },
      { index: 2, score: 1.2 },
      { index: 0, score: 0.4 },
    ]);
    expect(forward.results[0]).not.toHaveProperty('kept');

    const reversed = [...DOCUMENTS].reverse();
    const backward = await reranker.rank({ query: 'q', documents: reversed });
    expect(order(backward).map((index) => reversed[index])).toEqual(
      order(forward).map((index) => DOCUMENTS[index])
    );
  });

  it('keeps the rejecting Reranker contract: topK slices, usage is reported, failures reject', async () => {
    const server = await stub(scored(SCORES));
    const reranker = createSystemOneReranker({
      model: jev(server.url),
      rubric,
    });
    await expect(
      reranker.rerank({ query: 'q', documents: DOCUMENTS, topK: 2 })
    ).resolves.toEqual({
      results: [
        { index: 3, score: 2.9 },
        { index: 1, score: 2.6 },
      ],
      model: 'jev-1.13.0',
      usage: { inputTokens: 40, outputTokens: 4 },
    });
    await expect(
      reranker.rerank({ query: 'q', documents: ['a'], topK: 0 })
    ).resolves.toEqual({ results: [], model: 'jev-1.13.0', usage: null });
    expect(server.calls).toHaveLength(4);
    await expect(
      createSystemOneReranker({
        model: jev(server.url),
        rubric,
        maxDocuments: 1,
      }).rerank({ query: 'q', documents: ['a', 'b'] })
    ).rejects.toMatchObject({ failure: 'unsupported_question' });
  });

  it.each([
    [401, 'unauthorized'],
    [402, 'quota'],
  ])(
    'stops on HTTP %s (%s): the rest are never sent and nothing is re-issued',
    async (status, reason) => {
      const server = await stub(() => ({ status }));
      const reranker = createSystemOneReranker({
        model: jev(server.url, {}, 'key'),
        rubric,
        concurrency: 1,
      });
      const outcome = await reranker.rank({
        query: 'q',
        documents: DOCUMENTS,
      });
      expect(server.calls).toHaveLength(1);
      expect(outcome).toMatchObject({ status: 'skipped', reason });
      expect(order(outcome)).toEqual([0, 1, 2, 3]);
      expect(outcome.results.every((entry) => entry.score === null)).toBe(true);
      await expect(
        reranker.rerank({ query: 'q', documents: DOCUMENTS })
      ).rejects.toMatchObject({ failure: reason, status });
    }
  );

  it('re-issues an unscored document exactly once; still unscored is partial, recovered is ok', async () => {
    const failing = await stub((call) =>
      documentOf(call) === 'gamma' ? { status: 500 } : scored(SCORES)(call)
    );
    const partial = await createSystemOneReranker({
      model: jev(failing.url),
      rubric,
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(failing.calls.map(documentOf).filter((d) => d === 'gamma')).toEqual([
      'gamma',
      'gamma',
    ]);
    expect(failing.calls).toHaveLength(5);
    expect(partial).toMatchObject({
      status: 'skipped',
      reason: 'partial',
      usage: { calls: 5, inputTokens: 30, estimatedTokens: 0 },
    });
    expect(order(partial)).toEqual([0, 1, 2, 3]);
    await expect(
      createSystemOneReranker({ model: jev(failing.url), rubric }).rerank({
        query: 'q',
        documents: DOCUMENTS,
      })
    ).rejects.toMatchObject({ failure: 'malformed_response' });

    let refused = 0;
    const flaky = await stub((call) =>
      documentOf(call) === 'gamma' && refused++ === 0
        ? { status: 503 }
        : scored(SCORES)(call)
    );
    const recovered = await createSystemOneReranker({
      model: jev(flaky.url),
      rubric,
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(recovered.status).toBe('ok');
    expect(order(recovered)).toEqual([3, 1, 2, 0]);
    expect(flaky.calls).toHaveLength(5);
  });

  it('every document failing reports the documents’ own failure, not partial', async () => {
    const server = await stub(() => ({ status: 429 }));
    const outcome = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
    }).rank({ query: 'q', documents: ['a', 'b'] });
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'rate_limited',
    });
    // A 429 is the host's, not one document's: nothing is re-issued into it.
    expect(server.calls).toHaveLength(2);
  });

  it('returns at its deadline as timeout and charges the abandoned request at the estimate', async () => {
    const server = await stub((call) =>
      documentOf(call) === 'gamma' ? 'hang' : scored(SCORES)(call)
    );
    const reranker = createSystemOneReranker({
      model: jev(server.url),
      rubric,
      timeoutMs: 5_000,
    });
    const started = performance.now();
    const outcome = await reranker.rank({
      query: 'q',
      documents: DOCUMENTS,
      deadlineMs: 300,
    });
    expect(performance.now() - started).toBeLessThan(1_500);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      usage: { calls: 4, inputTokens: 30, estimatedTokens: 600 },
    });
    expect(order(outcome)).toEqual([0, 1, 2, 3]);
    await expect(
      reranker.rerank({ query: 'q', documents: DOCUMENTS, timeoutMs: 300 })
    ).rejects.toMatchObject({ failure: 'timeout' });
  });

  it('a caller abort mid-flight is aborted, not timeout', async () => {
    const server = await stub(() => 'hang');
    const controller = new AbortController();
    const pending = createSystemOneReranker({
      model: jev(server.url),
      rubric,
    }).rank({ query: 'q', documents: ['a'], signal: controller.signal });
    await new Promise((resolve) => setTimeout(resolve, 50));
    controller.abort();
    expect(await pending).toMatchObject({
      status: 'skipped',
      reason: 'aborted',
    });
  });

  it('with a pin, a response that does not echo it is pin_mismatch and stops the step', async () => {
    const server = await stub(scored(SCORES, 'jev-1.14.0'));
    const reranker = createSystemOneReranker({
      model: jev(server.url),
      rubric,
      concurrency: 1,
      pin: { model: 'jev-1.13.0', threshold: 2 },
    });
    const outcome = await reranker.rank({ query: 'q', documents: DOCUMENTS });
    expect(server.calls).toHaveLength(1);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'pin_mismatch',
      model: 'jev-1.14.0',
      pinVerified: false,
    });
    expect(outcome).not.toHaveProperty('threshold');
    expect(outcome.results.some((entry) => 'kept' in entry)).toBe(false);
    await expect(
      reranker.rerank({ query: 'q', documents: DOCUMENTS })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
  });

  it('refuses a pin that is not the configured model without a request', async () => {
    const server = await stub(scored(SCORES));
    const outcome = await createSystemOneReranker({
      model: jev(server.url, { model: 'jev-latest' }),
      rubric,
      pin: { model: 'jev-1.13.0' },
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'pin_mismatch',
      pinVerified: false,
    });
    expect(outcome.attempts[0].status).toBe('skipped');
    expect(server.calls).toHaveLength(0);
  });

  it('sets kept only with a calibrated threshold on a verified pin', async () => {
    const server = await stub(scored(SCORES));
    const pinned = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      pin: { model: 'jev-1.13.0', threshold: 2 },
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(pinned).toMatchObject({
      status: 'ok',
      pinVerified: true,
      threshold: 2,
    });
    expect(pinned.results).toEqual([
      { index: 3, score: 2.9, kept: true },
      { index: 1, score: 2.6, kept: true },
      { index: 2, score: 1.2, kept: false },
      { index: 0, score: 0.4, kept: false },
    ]);
    const uncalibrated = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      pin: { model: 'jev-1.13.0' },
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(uncalibrated).toMatchObject({ status: 'ok', pinVerified: true });
    expect(uncalibrated).not.toHaveProperty('threshold');
    expect(uncalibrated.results.some((entry) => 'kept' in entry)).toBe(false);
  });

  it('caches scores by document text; a cache hit is not sent and another model is never cached', async () => {
    const server = await stub(scored({ ...SCORES, epsilon: 1 }));
    const cache = createScoreCache();
    const reranker = createSystemOneReranker({
      model: jev(server.url),
      rubric,
      cache,
    });
    await reranker.rank({ query: 'q', documents: DOCUMENTS });
    expect(server.calls).toHaveLength(4);
    const again = await reranker.rank({
      query: 'q',
      documents: ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
    });
    expect(server.calls).toHaveLength(5);
    expect(documentOf(server.calls[4])).toBe('epsilon');
    expect(again.usage).toMatchObject({ calls: 1, cached: 4 });

    const other = await stub(scored(SCORES, 'jev-1.14.0'));
    const otherCache = createScoreCache();
    await createSystemOneReranker({
      model: jev(other.url),
      rubric,
      cache: otherCache,
    }).rank({ query: 'q', documents: ['alpha'] });
    expect(otherCache.size).toBe(0);
  });

  it('never has more requests in flight than its concurrency', async () => {
    const server = await stub((call) => ({
      ...(scored(SCORES)(call) as Json),
      delayMs: 20,
    }));
    const outcome = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      concurrency: 2,
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(outcome.status).toBe('ok');
    expect(server.maxConcurrent).toBe(2);
  });

  it('an open breaker is circuit_open and sends nothing', async () => {
    const server = await stub(() => ({ status: 401 }));
    const breaker = createBreaker();
    const reranker = createSystemOneReranker({
      model: withBreaker(jev(server.url, {}, 'key'), breaker),
      rubric,
      concurrency: 1,
    });
    expect(
      (await reranker.rank({ query: 'q', documents: ['a', 'b'] })).reason
    ).toBe('unauthorized');
    expect(breaker.open).toBe(true);
    expect(
      (await reranker.rank({ query: 'q', documents: ['a', 'b'] })).reason
    ).toBe('circuit_open');
    expect(server.calls).toHaveLength(1);

    const wide = await createSystemOneReranker({
      model: withBreaker(jev(server.url, {}, 'key'), breaker),
      rubric,
      concurrency: 4,
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(wide).toMatchObject({
      reason: 'circuit_open',
      usage: { calls: 0, estimatedTokens: 0 },
    });
    expect(wide.attempts[0]).toMatchObject({
      status: 'skipped',
      reason: 'circuit_open',
    });
    const batch = await createSystemOneReranker({
      model: withBreaker(jev(server.url, {}, 'key'), breaker),
      rubric,
      mode: 'batch',
    }).rank({ query: 'q', documents: DOCUMENTS });
    expect(batch).toMatchObject({
      reason: 'circuit_open',
      usage: { calls: 0, estimatedTokens: 0 },
    });
    expect(batch.attempts[0].status).toBe('skipped');
    expect(server.calls).toHaveLength(1);
  });

  it('treats an out-of-rubric score from a host cache as a miss', async () => {
    const server = await stub(scored(SCORES));
    const outcome = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      cache: {
        get: (key) => ({ alpha: 99, beta: -5 })[key.document],
        set: () => {},
      },
    }).rank({ query: 'q', documents: ['alpha', 'beta'] });
    expect(server.calls.map(documentOf).sort()).toEqual(['alpha', 'beta']);
    expect(outcome).toMatchObject({ status: 'ok', usage: { cached: 0 } });
    expect(outcome.results).toEqual([
      { index: 1, score: 2.6 },
      { index: 0, score: 0.4 },
    ]);
  });

  it('keys cached scores on a custom state, so rerankers judging in different states never share one', async () => {
    const server = await stub((call) => ({
      body: {
        model: call.body.model,
        answers: {
          relevance: {
            type: 'score',
            score: (call.body.state as Json).tenant === 'a' ? 0 : 2,
          },
        },
      },
    }));
    const cache = createScoreCache();
    const tenant = (name: string) => (query: string, document: string) => ({
      query,
      document,
      tenant: name,
    });
    const a = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      cache,
      state: tenant('a'),
    }).rank({ query: 'q', documents: ['d'] });
    const b = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      cache,
      state: tenant('b'),
    }).rank({ query: 'q', documents: ['d'] });
    expect(a.results[0].score).toBe(0);
    expect(b.results[0].score).toBe(2);
    expect(b.usage).toMatchObject({ calls: 1, cached: 0 });
    expect(server.calls).toHaveLength(2);
    const again = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
      cache,
      state: tenant('b'),
    }).rank({ query: 'q', documents: ['d'] });
    expect(again.usage).toMatchObject({ calls: 0, cached: 1 });
  });

  it('refuses a score cache bound that would unbound it', () => {
    for (const options of [
      { max: Number.NaN },
      { max: Infinity },
      { max: 0 },
      { max: 1.5 },
      { ttlMs: Number.NaN },
      { ttlMs: Infinity },
      { ttlMs: -1 },
    ]) {
      expect(() => createScoreCache(options)).toThrow(
        'invalid score cache configuration'
      );
    }
    const cache = createScoreCache({ max: 2 });
    const key = (document: string) => ({
      query: 'q',
      document,
      model: 'm',
      rubric: 'r',
      state: '',
    });
    cache.set(key('a'), 1);
    cache.set(key('b'), 1);
    cache.set(key('c'), 1);
    expect(cache.size).toBe(2);
    expect(cache.get(key('a'))).toBeUndefined();
  });

  it('never throws: a throwing model, invalid input, and an empty list are outcomes', async () => {
    const throwing: DecisionModel = {
      id: 'throws',
      model: 'm',
      decide: () => {
        throw new Error('secret provider text');
      },
    };
    const reranker = createSystemOneReranker({ model: throwing, rubric });
    const thrown = await reranker.rank({ query: 'q', documents: ['a'] });
    expect(thrown).toMatchObject({ status: 'skipped', reason: 'error' });
    expect(JSON.stringify(thrown)).not.toContain('secret');
    expect((await reranker.rank({ query: 'q', documents: [] })).reason).toBe(
      'empty'
    );
    expect(
      (
        await reranker.rank({
          query: '',
          documents: ['a'],
        })
      ).reason
    ).toBe('invalid_request');
    expect(
      (
        await reranker.rank({
          documents: 'nope',
        } as unknown as RankRequest)
      ).results
    ).toEqual([]);
  });

  it('reports a model that times out every request on its own clock as timeout, charged as abandoned', async () => {
    const slow: DecisionModel = {
      id: 'capped',
      model: 'jev-1.13.0',
      decide: () =>
        new Promise((_, reject) =>
          setTimeout(
            () =>
              reject(
                new DecisionError('timeout', 'decision model timed out', {
                  provider: 'capped',
                })
              ),
            20
          )
        ),
    };
    const reranker = createSystemOneReranker({
      model: slow,
      rubric,
      timeoutMs: 10_000,
    });
    const outcome = await reranker.rank({ query: 'q', documents: ['a', 'b'] });
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      usage: { calls: 4, estimatedTokens: 2_400 },
    });
    expect(order(outcome)).toEqual([0, 1]);
    await expect(
      reranker.rerank({ query: 'q', documents: ['a', 'b'] })
    ).rejects.toMatchObject({ failure: 'timeout' });
  });

  it('reports an oversized candidate set as too_many_documents in both modes, without a request', async () => {
    const server = await stub(scored(SCORES));
    for (const mode of ['document', 'batch'] as const) {
      const outcome = await createSystemOneReranker({
        model: jev(server.url),
        rubric,
        mode,
        maxDocuments: 2,
      }).rank({ query: 'q', documents: ['a', 'b', 'c'] });
      expect(outcome).toMatchObject({
        status: 'skipped',
        reason: 'too_many_documents',
      });
      expect(outcome.attempts.map((attempt) => attempt.status)).toEqual([
        'skipped',
      ]);
      expect(order(outcome)).toEqual([0, 1, 2]);
    }
    expect(server.calls).toHaveLength(0);
  });

  it('rejects document-mode options in batch mode', () => {
    const model = createDecisionModel(
      { ...decisionPreset('laya'), baseURL: 'http://127.0.0.1:1/v1/systemone' },
      undefined
    );
    expect(() =>
      createSystemOneReranker({
        model,
        rubric,
        mode: 'batch',
        pin: { model: 'jev-1.13.0' },
      })
    ).toThrow('invalid reranker configuration');
    expect(() =>
      createSystemOneReranker({
        model,
        rubric,
        mode: 'batch',
        estimatedTokensPerDocument: 100,
      })
    ).toThrow('invalid reranker configuration');
  });

  it('rejects a malformed timeout, concurrency, token estimate, or instructions instead of coercing it', () => {
    const { model } = fakeJev(() => 1);
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    for (const broken of [
      { estimatedTokensPerDocument: Number.NaN },
      { estimatedTokensPerDocument: -5 },
      { estimatedTokensPerDocument: Infinity },
      { timeoutMs: 0 },
      { timeoutMs: Number.NaN },
      // Fractions are rejected, not kept: the deadline wrapper refuses them on every call.
      { timeoutMs: 0.5 },
      { timeoutMs: (0.1 + 0.2) * 1000 },
      { mode: 'batch' as const, timeoutMs: 0.5 },
      { concurrency: 0 },
      { concurrency: 1.5 },
      { instructions: circular as never },
    ]) {
      expect(() =>
        createSystemOneReranker({ model, rubric, ...broken })
      ).toThrow('invalid reranker configuration');
    }
    expect(() =>
      createSystemOneReranker({ model, rubric, estimatedTokensPerDocument: 0 })
    ).not.toThrow();
  });

  it('counts its deadline from the call, so slow cache lookups spend it and nothing is sent after it', async () => {
    const { model, sent } = fakeJev(() => 2);
    const slowCache = {
      get: (): number | undefined => {
        spin(150);
        return undefined;
      },
      set: (): void => {},
    };
    const reranker = createSystemOneReranker({
      model,
      rubric,
      cache: slowCache,
    });
    const outcome = await reranker.rank({
      query: 'q',
      documents: ['a', 'b'],
      deadlineMs: 200,
    });
    expect(sent).toHaveLength(0);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      usage: { ...emptyUsage },
    });
    expect(outcome.attempts[0].status).toBe('skipped');
    await expect(
      reranker.rerank({ query: 'q', documents: ['a', 'b'], timeoutMs: 200 })
    ).rejects.toMatchObject({ failure: 'timeout' });

    const full = createScoreCache();
    const warm = createSystemOneReranker({ model, rubric, cache: full });
    expect(
      (await warm.rank({ query: 'q', documents: ['a', 'b'] })).status
    ).toBe('ok');
    const calls = sent.length;
    const slowHits = {
      get: (key: Parameters<typeof full.get>[0]): number | undefined => {
        spin(150);
        return full.get(key);
      },
      set: (): void => {},
    };
    const served = await createSystemOneReranker({
      model,
      rubric,
      cache: slowHits,
    }).rank({ query: 'q', documents: ['a', 'b'], deadlineMs: 200 });
    expect(served).toMatchObject({ status: 'ok', usage: { cached: 2 } });
    expect(sent).toHaveLength(calls);
  });

  it('sends no request once building a custom state has spent the deadline', async () => {
    const { model, sent } = fakeJev(() => 2, 100);
    const outcome = await createSystemOneReranker({
      model,
      rubric,
      state: (query, document) => {
        spin(150);
        return { query, document };
      },
    }).rank({ query: 'q', documents: ['a', 'b'], deadlineMs: 200 });
    expect(sent.map((request) => request.state)).toEqual([
      { query: 'q', document: 'a' },
    ]);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      usage: { calls: 1, estimatedTokens: 600 },
    });
  });

  it('counts nothing for a custom state that throws, since no request is sent', async () => {
    const { model, sent } = fakeJev(() => 2);
    const outcome = await createSystemOneReranker({
      model,
      rubric,
      state: () => {
        throw new Error('boom');
      },
    }).rank({ query: 'q', documents: ['a', 'b', 'c'] });
    expect(sent).toHaveLength(0);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'error',
      usage: emptyUsage,
    });
    expect(outcome.attempts).toEqual([
      expect.objectContaining({
        status: 'skipped',
        reason: 'error',
        usage: emptyUsage,
      }),
    ]);
    const partial = await createSystemOneReranker({
      model,
      rubric,
      state: (query, document) => {
        if (document === 'b') {
          throw new Error('boom');
        }
        return { query, document };
      },
    }).rank({ query: 'q', documents: ['a', 'b', 'c'] });
    expect(sent).toHaveLength(2);
    expect(partial).toMatchObject({
      status: 'skipped',
      reason: 'partial',
      usage: { calls: 2, inputTokens: 20, estimatedTokens: 0 },
    });
  });

  it('ignores a score cache whose methods reject, without an unhandled rejection', async () => {
    const { model, sent } = fakeJev(() => 2);
    const cache = {
      get: (() => Promise.reject(new Error('get'))) as unknown as (
        key: unknown
      ) => number | undefined,
      set: async (): Promise<void> => {
        throw new Error('set');
      },
    };
    let outcome: RerankOutcome | undefined;
    const unhandled = await unhandledDuring(async () => {
      outcome = await createSystemOneReranker({ model, rubric, cache }).rank({
        query: 'q',
        documents: ['a', 'b'],
      });
    });
    expect(unhandled).toEqual([]);
    expect(sent).toHaveLength(2);
    expect(outcome).toMatchObject({
      status: 'ok',
      usage: { calls: 2, cached: 0 },
    });
  });

  it('snapshots structured instructions, so a caller mutating them never splits what is asked from what is cached', async () => {
    const instructions = { strict: false };
    const { model, sent } = fakeJev((request) =>
      (request.questions.relevance.instructions as { strict: boolean }).strict
        ? 0
        : 2
    );
    const reranker = createSystemOneReranker({
      model,
      rubric,
      instructions,
      cache: createScoreCache(),
    });
    await reranker.rank({ query: 'q', documents: ['d'] });
    instructions.strict = true;
    const outcome = await reranker.rank({ query: 'q', documents: ['d2'] });
    expect(
      sent.map((request) => request.questions.relevance.instructions)
    ).toEqual([{ strict: false }, { strict: false }]);
    expect(outcome.results[0].score).toBe(2);
  });

  it('snapshots a custom state with its cache key, so a caller mutating it after rank never caches one state under another', async () => {
    const { model, sent } = fakeJev((request) =>
      (request.state as { variant: string }).variant === 'A' ? 2 : 0
    );
    const shared = { variant: 'A' };
    let fresh = false;
    const reranker = createSystemOneReranker({
      model,
      rubric,
      cache: createScoreCache(),
      state: () => (fresh ? { variant: 'A' } : shared),
    });
    const pending = reranker.rank({ query: 'q', documents: ['d'] });
    shared.variant = 'B';
    const first = await pending;
    expect(sent.map((request) => request.state)).toEqual([{ variant: 'A' }]);
    expect(first.results[0].score).toBe(2);
    fresh = true;
    const second = await reranker.rank({ query: 'q', documents: ['d'] });
    expect(sent).toHaveLength(1);
    expect(second).toMatchObject({ usage: { calls: 0, cached: 1 } });
    expect(second.results[0].score).toBe(2);
  });
});

/** Cohere-shaped rerank results for `documents`, scored by `scores`. */
function relevance(
  scores: Record<string, number>,
  extra: Json = {}
): (call: Call) => Reply {
  return (call) => ({
    body: {
      model: call.body.model,
      results: (call.body.documents as string[])
        .map((document, index) => ({
          index,
          relevance_score: scores[document],
        }))
        .sort((a, b) => b.relevance_score - a.relevance_score),
      meta: { billed_units: { search_units: 1 }, tokens: { input_tokens: 80 } },
      ...extra,
    },
  });
}

const RELEVANCE = { alpha: 0.1, beta: 0.7, gamma: 0.3, delta: 0.9 };

describe('System One usage counts transport retries', () => {
  /** A Jev model with the transport's own retries and no backoff wait. */
  function retrying(url: string, apiKey?: DecisionCredential): DecisionModel {
    return createHttpDecisionModel({
      providerId: 'jev',
      endpoint: `${url}/v1/systemone`,
      model: 'jev-1.13.0',
      dialect: 'systemone',
      requiresAuth: apiKey !== undefined,
      apiKey,
      maxRetries: 2,
      sleep: () => Promise.resolve(),
    });
  }

  /** Fails each document's first `failures` requests with `status`, then scores it. */
  function flaky(failures: number, status = 503): (call: Call) => Reply {
    const seen = new Map<string, number>();
    return (call) => {
      const document = documentOf(call);
      const nth = seen.get(document) ?? 0;
      seen.set(document, nth + 1);
      return nth < failures ? { status } : scored(SCORES)(call);
    };
  }

  it('document mode counts every retried request, not one per decide()', async () => {
    const server = await stub(flaky(2));
    const outcome = await createSystemOneReranker({
      model: retrying(server.url),
      rubric,
    }).rank({ query: 'q', documents: ['alpha', 'beta'] });
    expect(server.calls).toHaveLength(6);
    expect(outcome).toMatchObject({ status: 'ok' });
    expect(outcome.usage).toMatchObject({
      calls: 6,
      inputTokens: 20,
      estimatedTokens: 0,
    });
    expect(outcome.attempts[0].usage.calls).toBe(6);
  });

  it('document mode counts the re-send after a 401 refresh', async () => {
    const server = await stub((call) =>
      call.authorization === 'Bearer stale'
        ? { status: 401 }
        : scored(SCORES)(call)
    );
    const outcome = await createSystemOneReranker({
      model: retrying(server.url, async ({ refresh }) =>
        refresh ? 'fresh' : 'stale'
      ),
      rubric,
    }).rank({ query: 'q', documents: ['alpha'] });
    expect(server.calls).toHaveLength(2);
    expect(outcome).toMatchObject({ status: 'ok' });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  it('document mode counts the re-send after a recheck that follows a second 401', async () => {
    const server = await stub((call) =>
      call.authorization === 'Bearer newer'
        ? scored(SCORES)(call)
        : { status: 401 }
    );
    let cached = 'stale';
    const outcome = await createSystemOneReranker({
      model: retrying(server.url, async ({ refresh }) => {
        if (refresh) {
          /** Another call has `fresh` refused and caches `newer` while this request is in flight. */
          cached = 'newer';
          return 'fresh';
        }
        return cached;
      }),
      rubric,
    }).rank({ query: 'q', documents: ['alpha'] });
    expect(server.calls.map((call) => call.authorization)).toEqual([
      'Bearer stale',
      'Bearer fresh',
      'Bearer newer',
    ]);
    expect(outcome).toMatchObject({ status: 'ok' });
    expect(outcome.usage).toMatchObject({ calls: 3, estimatedTokens: 0 });
  });

  it('estimates only the retried request a deadline abandoned', async () => {
    const seen = new Map<string, number>();
    const server = await stub((call) => {
      const nth = seen.get(documentOf(call)) ?? 0;
      seen.set(documentOf(call), nth + 1);
      return nth === 0 ? { status: 503 } : 'hang';
    });
    const outcome = await createSystemOneReranker({
      model: retrying(server.url),
      rubric,
      timeoutMs: 300,
    }).rank({ query: 'q', documents: ['alpha'] });
    expect(server.calls).toHaveLength(2);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 600 });
  });

  it('batch mode counts every retried request', async () => {
    const answer: Reply = {
      body: {
        model: 'jev-1.13.0',
        answers: {
          d0: { type: 'score', score: 2 },
          d1: { type: 'score', score: 1 },
        },
        usage: { input_tokens: 30, output_tokens: 2 },
      },
    };
    const server = await stub((_call, nth) =>
      nth === 0 ? { status: 503 } : answer
    );
    const outcome = await createSystemOneReranker({
      model: retrying(server.url),
      rubric,
      mode: 'batch',
    }).rank({ query: 'q', documents: ['alpha', 'beta'] });
    expect(server.calls).toHaveLength(2);
    expect(outcome).toMatchObject({ status: 'ok' });
    expect(outcome.usage).toMatchObject({ calls: 2, inputTokens: 30 });
  });
});

describe('cross-encoder ranking', () => {
  it('posts { model, query, documents } once with the bearer key and ranks by relevance_score', async () => {
    const server = await stub(relevance(RELEVANCE));
    const reranker = createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'qwen3-reranker-8b',
      apiKey: 'gateway-key',
    });
    expect(reranker).toMatchObject({
      id: 'cross-encoder:qwen3-reranker-8b',
      kind: 'cross-encoder',
    });
    const outcome = await reranker.rank({ query: 'q', documents: DOCUMENTS });
    expect(server.calls).toHaveLength(1);
    expect(server.calls[0]).toEqual({
      path: '/rerank',
      body: { model: 'qwen3-reranker-8b', query: 'q', documents: DOCUMENTS },
      authorization: 'Bearer gateway-key',
    });
    expect(outcome).toMatchObject({
      status: 'ok',
      provider: 'cross-encoder',
      model: 'qwen3-reranker-8b',
      usage: { calls: 1, inputTokens: 80, searchUnits: 1 },
    });
    expect(outcome.results).toEqual([
      { index: 3, score: 0.9 },
      { index: 1, score: 0.7 },
      { index: 2, score: 0.3 },
      { index: 0, score: 0.1 },
    ]);
  });

  it.each([
    [
      'a duplicated index',
      {
        results: [
          { index: 0, relevance_score: 0.5 },
          { index: 0, relevance_score: 0.4 },
        ],
      },
      'invalid_results',
    ],
    [
      'a missing document',
      { results: [{ index: 1, relevance_score: 0.5 }] },
      'invalid_results',
    ],
    [
      'an index out of range',
      {
        results: [
          { index: 0, relevance_score: 0.5 },
          { index: 2, relevance_score: 0.4 },
        ],
      },
      'invalid_results',
    ],
    [
      'a non-finite score',
      {
        results: [
          { index: 0, relevance_score: 0.5 },
          { index: 1, relevance_score: 'high' },
        ],
      },
      'invalid_results',
    ],
  ])(
    'rejects a response with %s as invalid_results, in identity order',
    async (_label, body, reason) => {
      const server = await stub(() => ({ body }));
      const outcome = await createCrossEncoderReranker({
        url: `${server.url}/rerank`,
        model: 'm',
        apiKey: 'k',
      }).rank({ query: 'q', documents: ['a', 'b'] });
      expect(outcome).toMatchObject({ status: 'skipped', reason });
      expect(outcome.results).toEqual([
        { index: 0, score: null },
        { index: 1, score: null },
      ]);
      expect(outcome.attempts[0].status).toBe('failed');
    }
  );

  it.each([
    [{ status: 401 }, 'unauthorized'],
    [{ status: 402 }, 'quota'],
    [{ status: 429 }, 'rate_limited'],
    [{ status: 500 }, 'error'],
    [{ raw: 'not json' }, 'bad_response'],
  ] as const)('reports %j as %s without throwing', async (reply, reason) => {
    const server = await stub(() => reply);
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'm',
      apiKey: 'k',
    }).rank({ query: 'q', documents: ['a'] });
    expect(outcome).toMatchObject({ status: 'skipped', reason });
    expect(server.calls).toHaveLength(1);
  });

  it('skips without a request when the url or key is missing', async () => {
    expect(
      (
        await createCrossEncoderReranker({ model: 'm', apiKey: 'k' }).rank({
          query: 'q',
          documents: ['a'],
        })
      ).reason
    ).toBe('no_endpoint');
    const server = await stub(relevance({ a: 1 }));
    const keyless = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'm',
    }).rank({ query: 'q', documents: ['a'] });
    expect(keyless).toMatchObject({ reason: 'no_api_key' });
    expect(keyless.attempts[0].status).toBe('skipped');
    expect(server.calls).toHaveLength(0);
    const local = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'm',
      requiresAuth: false,
    }).rank({ query: 'q', documents: ['a'] });
    expect(local.status).toBe('ok');
    expect(server.calls[0].authorization).toBeUndefined();
  });

  it('re-mints a refused token once', async () => {
    const server = await stub((call) =>
      call.authorization === 'Bearer fresh'
        ? relevance({ a: 1 })(call)
        : { status: 401 }
    );
    const minted: boolean[] = [];
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'm',
      apiKey: async ({ refresh }) => {
        minted.push(refresh);
        return refresh ? 'fresh' : 'stale';
      },
    }).rank({ query: 'q', documents: ['a'] });
    expect(outcome.status).toBe('ok');
    expect(minted).toEqual([false, true]);
    expect(outcome.usage.calls).toBe(2);
  });

  it('rejects a fractional timeout at construction instead of failing every ranking', async () => {
    const server = await stub(relevance(RELEVANCE));
    const base = { url: `${server.url}/rerank`, model: 'm', apiKey: 'k' };
    for (const timeoutMs of [0.5, (0.1 + 0.2) * 1000]) {
      expect(() => createCrossEncoderReranker({ ...base, timeoutMs })).toThrow(
        'invalid reranker configuration'
      );
    }
    const outcome = await createCrossEncoderReranker({
      ...base,
      timeoutMs: 1100,
    }).rank({ query: 'q', documents: ['alpha', 'beta'] });
    expect(outcome.status).toBe('ok');
    expect(server.calls).toHaveLength(1);
  });

  it('rejects a token estimate that is not a finite count instead of charging nothing', () => {
    for (const estimatedTokensPerDocument of [
      Number.NaN,
      -5,
      Infinity,
      Number('abc'),
    ]) {
      expect(() =>
        createCrossEncoderReranker({
          url: 'http://127.0.0.1:1/rerank',
          model: 'm',
          apiKey: 'k',
          estimatedTokensPerDocument,
        })
      ).toThrow('invalid reranker configuration');
    }
  });

  it('estimates search units from the document count even with no token estimate', async () => {
    expect(estimatedUsage('cross-encoder', 4, 0)).toEqual({
      estimatedTokens: 0,
      estimatedSearchUnits: 1,
    });
    const server = await stub(() => 'hang');
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'm',
      apiKey: 'k',
      estimatedTokensPerDocument: 0,
    }).rank({ query: 'q', documents: DOCUMENTS, deadlineMs: 200 });
    expect(outcome.usage).toMatchObject({
      calls: 1,
      estimatedTokens: 0,
      estimatedSearchUnits: 1,
    });
    expect(
      rerankCost(outcome, {
        'cross-encoder': { '*': { perSearchUnit: 0.002 } },
      })
    ).toEqual({ usd: 0.002, unpriced: [] });
  });

  it('returns at its deadline as timeout, charged one estimated request', async () => {
    const server = await stub(() => 'hang');
    const started = performance.now();
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'm',
      apiKey: 'k',
    }).rank({ query: 'q', documents: DOCUMENTS, deadlineMs: 200 });
    expect(performance.now() - started).toBeLessThan(1_500);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      usage: { calls: 1, estimatedTokens: 800, estimatedSearchUnits: 1 },
    });
  });
});

/** A reranker that answers whatever its test says, recording what it was asked. */
function scripted(
  kind: OutcomeReranker['kind'],
  answer: (request: RankRequest) => Promise<RerankOutcome> | RerankOutcome
): OutcomeReranker & { requests: RankRequest[] } {
  const requests: RankRequest[] = [];
  return {
    id: `scripted:${kind}`,
    kind,
    model: `${kind}-model`,
    requests,
    rank: async (request) => {
      requests.push(request);
      return answer(request);
    },
  };
}

describe('withFallback', () => {
  it('tries steps in order: a refused System One step degrades to the cross-encoder with the first reason', async () => {
    const server = await stub((call) =>
      call.path === '/v1/systemone'
        ? { status: 401 }
        : relevance(RELEVANCE)(call)
    );
    const chain = withFallback([
      {
        reranker: createSystemOneReranker({
          model: jev(server.url, {}, 'key'),
          rubric,
          concurrency: 1,
          pin: { model: 'jev-1.13.0', threshold: 2 },
        }),
        deadlineMs: 900,
      },
      {
        reranker: createCrossEncoderReranker({
          url: `${server.url}/rerank`,
          model: 'qwen3-reranker-8b',
          apiKey: 'key',
        }),
        deadlineMs: 500,
      },
      { reranker: noneReranker() },
    ]);
    const outcome = await chain.rank({ query: 'q', documents: DOCUMENTS });
    expect(server.calls.map((call) => call.path)).toEqual([
      '/v1/systemone',
      '/rerank',
    ]);
    expect(outcome).toMatchObject({
      status: 'degraded',
      reason: 'unauthorized',
      provider: 'cross-encoder',
      model: 'qwen3-reranker-8b',
      usage: { calls: 2, inputTokens: 80, searchUnits: 1 },
    });
    expect(outcome).not.toHaveProperty('threshold');
    expect(outcome).not.toHaveProperty('pinVerified');
    expect(outcome.results.some((entry) => 'kept' in entry)).toBe(false);
    expect(
      outcome.attempts.map((attempt) => [attempt.provider, attempt.status])
    ).toEqual([
      ['systemone', 'failed'],
      ['cross-encoder', 'ok'],
    ]);
    expect(order(outcome)).toEqual([3, 1, 2, 0]);
  });

  it('step 0 answering is ok and carries its kept, threshold, and pin', async () => {
    const server = await stub(scored(SCORES));
    const outcome = await withFallback([
      {
        reranker: createSystemOneReranker({
          model: jev(server.url),
          rubric,
          pin: { model: 'jev-1.13.0', threshold: 2 },
        }),
        deadlineMs: 2_000,
      },
      { reranker: noneReranker() },
    ]).rank({ query: 'q', documents: DOCUMENTS });
    expect(outcome).toMatchObject({
      status: 'ok',
      reason: null,
      pinVerified: true,
      threshold: 2,
    });
    expect(outcome.results[0]).toEqual({ index: 3, score: 2.9, kept: true });
  });

  it('a hung first step that ignores its abort still lets the next step answer within both deadlines', async () => {
    const server = await stub(relevance(RELEVANCE));
    const hung = scripted('systemone', () => new Promise(() => {}));
    const started = performance.now();
    const outcome = await withFallback([
      { reranker: hung, deadlineMs: 150 },
      {
        reranker: createCrossEncoderReranker({
          url: `${server.url}/rerank`,
          model: 'm',
          apiKey: 'k',
        }),
        deadlineMs: 500,
      },
    ]).rank({ query: 'q', documents: DOCUMENTS });
    expect(performance.now() - started).toBeLessThan(150 + 50 + 500);
    expect(hung.requests[0].signal?.aborted).toBe(true);
    expect(hung.requests[0].deadlineMs).toBe(150);
    expect(outcome).toMatchObject({ status: 'degraded', reason: 'timeout' });
    expect(outcome.attempts[0]).toMatchObject({
      provider: 'systemone',
      status: 'failed',
      reason: 'timeout',
      usage: { calls: 4, estimatedTokens: 2_400 },
    });
    expect(outcome.usage).toMatchObject({
      calls: 5,
      estimatedTokens: 2_400,
      inputTokens: 80,
    });
  });

  it('a cooperative System One step reports its own timeout and spend through the chain', async () => {
    const server = await stub((call) =>
      documentOf(call) === 'delta' ? 'hang' : scored(SCORES)(call)
    );
    const outcome = await withFallback([
      {
        reranker: createSystemOneReranker({ model: jev(server.url), rubric }),
        deadlineMs: 250,
      },
      { reranker: noneReranker({ reason: 'no_api_key' }) },
    ]).rank({ query: 'q', documents: DOCUMENTS });
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      provider: 'none',
      usage: { calls: 4, inputTokens: 30, estimatedTokens: 600 },
    });
    expect(outcome.attempts[0]).toMatchObject({
      provider: 'systemone',
      status: 'failed',
      reason: 'timeout',
    });
    expect(order(outcome)).toEqual([0, 1, 2, 3]);
  });

  it.each([
    [
      'a partial permutation',
      { results: [{ index: 0, score: 1 }], status: 'ok', attempts: [] },
    ],
    [
      'a duplicated index',
      {
        results: [
          { index: 1, score: 1 },
          { index: 1, score: 0 },
        ],
        status: 'ok',
        attempts: [],
      },
    ],
    [
      'an unknown status',
      {
        results: [
          { index: 1, score: 1 },
          { index: 0, score: 0 },
        ],
        status: 'great',
        attempts: [],
      },
    ],
    ...(
      [
        [
          'unsorted scores',
          {
            results: [
              { index: 0, score: 0.1 },
              { index: 1, score: 0.9 },
            ],
          },
        ],
        [
          'a null score before a number',
          {
            results: [
              { index: 0, score: null },
              { index: 1, score: 0.9 },
            ],
          },
        ],
        [
          'tied scores out of input order',
          {
            results: [
              { index: 1, score: 1 },
              { index: 0, score: 1 },
            ],
          },
        ],
        [
          'a non-boolean kept',
          {
            results: [
              { index: 1, score: 1, kept: 'yes' },
              { index: 0, score: 0, kept: false },
            ],
          },
        ],
        [
          'an attempt without usage',
          {
            attempts: [
              {
                provider: 'cross-encoder',
                model: 'm',
                status: 'ok',
                reason: null,
                durationMs: 1,
              },
            ],
          },
        ],
        [
          'a null score in an answer',
          {
            results: [
              { index: 1, score: 1 },
              { index: 0, score: null },
            ],
          },
        ],
        [
          'a null score in a degraded answer',
          {
            status: 'degraded',
            reason: 'partial',
            results: [
              { index: 1, score: 1 },
              { index: 0, score: null },
            ],
          },
        ],
        [
          'kept without a threshold',
          {
            results: [
              { index: 1, score: 1, kept: true },
              { index: 0, score: 0, kept: false },
            ],
          },
        ],
        [
          'kept flags contradicting their threshold',
          {
            threshold: 0.5,
            results: [
              { index: 1, score: 1, kept: false },
              { index: 0, score: 0, kept: true },
            ],
          },
        ],
        [
          'a threshold without kept flags',
          {
            threshold: 0.5,
            results: [
              { index: 1, score: 1 },
              { index: 0, score: 0 },
            ],
          },
        ],
        [
          'a threshold with kept on only some entries',
          {
            threshold: 0.5,
            results: [
              { index: 1, score: 1, kept: true },
              { index: 0, score: 0 },
            ],
          },
        ],
        ['an unknown provider', { provider: 'bogus' }],
        ['a non-string reason', { reason: 7 }],
        ['an answer with a reason', { reason: 'partial' }],
        ['a degraded answer without a reason', { status: 'degraded' }],
        [
          'a degraded answer with an empty reason',
          { status: 'degraded', reason: '' },
        ],
        ['a skipped outcome without a reason', { status: 'skipped' }],
        [
          'a skipped outcome with an empty reason',
          { status: 'skipped', reason: '' },
        ],
        [
          'an ok attempt with a reason',
          {
            attempts: [
              {
                provider: 'cross-encoder',
                model: 'm',
                status: 'ok',
                reason: 'partial',
                durationMs: 1,
                usage: { ...emptyUsage },
              },
            ],
          },
        ],
        [
          'a failed attempt without a reason',
          {
            attempts: [
              {
                provider: 'cross-encoder',
                model: 'm',
                status: 'failed',
                reason: null,
                durationMs: 1,
                usage: { ...emptyUsage },
              },
            ],
          },
        ],
        [
          'a skipped attempt with an empty reason',
          {
            attempts: [
              {
                provider: 'cross-encoder',
                model: 'm',
                status: 'skipped',
                reason: '',
                durationMs: 1,
                usage: { ...emptyUsage },
              },
            ],
          },
        ],
        [
          'a skipped outcome with scores',
          { status: 'skipped', reason: 'no_api_key' },
        ],
        [
          'a skipped outcome with an ok attempt',
          {
            status: 'skipped',
            reason: 'no_api_key',
            results: [
              { index: 0, score: null },
              { index: 1, score: null },
            ],
            attempts: [
              {
                provider: 'cross-encoder',
                model: 'm',
                status: 'ok',
                reason: null,
                durationMs: 1,
                usage: { ...emptyUsage },
              },
            ],
          },
        ],
        [
          'an answer with no ok attempt',
          {
            attempts: [
              {
                provider: 'cross-encoder',
                model: 'm',
                status: 'failed',
                reason: 'timeout',
                durationMs: 1,
                usage: { ...emptyUsage },
              },
            ],
          },
        ],
        ['a non-numeric threshold', { threshold: 'x' }],
        ['a non-boolean pinVerified', { pinVerified: 'yes' }],
      ] as const
    ).map(([label, broken]): [string, Record<string, unknown>] => [
      label,
      {
        results: [
          { index: 1, score: 1 },
          { index: 0, score: 0 },
        ],
        provider: 'cross-encoder',
        model: 'm',
        status: 'ok',
        reason: null,
        attempts: [],
        durationMs: 0,
        usage: { ...emptyUsage },
        ...broken,
      },
    ]),
  ])(
    'treats a step returning %s as invalid_results and moves on',
    async (_label, garbage) => {
      const broken = scripted(
        'cross-encoder',
        () => garbage as unknown as RerankOutcome
      );
      const backup = scripted('bm25', () => ({
        results: [
          { index: 1, score: 3 },
          { index: 0, score: 1 },
        ],
        provider: 'bm25',
        model: 'okapi',
        status: 'ok',
        reason: null,
        attempts: [],
        durationMs: 0,
        usage: { ...emptyUsage },
      }));
      const outcome = await withFallback([
        { reranker: broken },
        { reranker: backup },
      ]).rank({ query: 'q', documents: ['a', 'b'] });
      expect(outcome).toMatchObject({
        status: 'degraded',
        reason: 'invalid_results',
        provider: 'bm25',
      });
      expect(outcome.attempts[0]).toMatchObject({
        provider: 'cross-encoder',
        status: 'failed',
        reason: 'invalid_results',
      });
      expect(order(outcome)).toEqual([1, 0]);
    }
  );

  it('never answers degraded without a reason, and an empty none reason is disabled', async () => {
    const degraded = scripted('cross-encoder', () => ({
      results: [
        { index: 1, score: 1 },
        { index: 0, score: 0 },
      ],
      provider: 'cross-encoder',
      model: 'm',
      status: 'degraded',
      reason: null,
      attempts: [],
      durationMs: 0,
      usage: { ...emptyUsage },
    }));
    const alone = await withFallback([{ reranker: degraded }]).rank({
      query: 'q',
      documents: ['a', 'b'],
    });
    expect(alone).toMatchObject({
      status: 'skipped',
      reason: 'invalid_results',
    });
    const nested = await withFallback([
      { reranker: withFallback([{ reranker: degraded }]) },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(nested).toMatchObject({
      status: 'skipped',
      reason: 'invalid_results',
    });
    const answered = await withFallback([
      {
        reranker: scripted('cross-encoder', () => ({
          results: [
            { index: 1, score: 1 },
            { index: 0, score: 0 },
          ],
          provider: 'cross-encoder',
          model: 'm',
          status: 'degraded',
          reason: 'partial',
          attempts: [],
          durationMs: 0,
          usage: { ...emptyUsage },
        })),
      },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(answered).toMatchObject({ status: 'degraded', reason: 'partial' });
    // The step answered, so its own attempt is `ok`, and an `ok` attempt has no reason.
    expect(answered.attempts).toEqual([
      expect.objectContaining({ status: 'ok', reason: null }),
    ]);
    const none = await withFallback([
      { reranker: noneReranker({ reason: '' }) },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(none).toMatchObject({ status: 'skipped', reason: 'disabled' });
    expect(none.attempts).toEqual([
      expect.objectContaining({ status: 'skipped', reason: 'disabled' }),
    ]);
  });

  it('a gate skips a step without a request, none never answers, and nothing answering is skipped with the first reason', async () => {
    const server = await stub(scored(SCORES));
    const systemone = createSystemOneReranker({
      model: jev(server.url),
      rubric,
    });
    const seen: string[] = [];
    const outcome = await withFallback(
      [
        { reranker: systemone, deadlineMs: 900 },
        {
          reranker: scripted('cross-encoder', () =>
            Promise.reject(new Error('x'))
          ),
        },
        { reranker: noneReranker({ reason: 'no_api_key' }) },
      ],
      {
        gate: (reranker) => (reranker.kind === 'systemone' ? 'backoff' : null),
        onAttempt: (reranker) => {
          seen.push(reranker.kind);
          throw new Error('bookkeeping failure');
        },
      }
    ).rank({ query: 'q', documents: ['a', 'b'] });
    expect(server.calls).toHaveLength(0);
    expect(seen).toEqual(['cross-encoder', 'none']);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'backoff',
      provider: 'none',
    });
    expect(
      outcome.attempts.map((attempt) => [attempt.status, attempt.reason])
    ).toEqual([
      ['skipped', 'backoff'],
      ['failed', 'error'],
      ['skipped', 'no_api_key'],
    ]);
    expect(outcome.attempts[2]).toMatchObject({
      provider: 'none',
      usage: emptyUsage,
    });
    expect(order(outcome)).toEqual([0, 1]);
  });

  it('passes through kept flags that match their threshold', async () => {
    const outcome = await withFallback([
      {
        reranker: scripted('systemone', () => ({
          results: [
            { index: 1, score: 3, kept: true },
            { index: 0, score: 0.5, kept: true },
            { index: 2, score: 0.4, kept: false },
          ],
          provider: 'systemone',
          model: 'jev-1.13.0',
          status: 'ok',
          reason: null,
          attempts: [],
          durationMs: 0,
          usage: { ...emptyUsage },
          threshold: 0.5,
        })),
      },
    ]).rank({ query: 'q', documents: ['a', 'b', 'c'] });
    expect(outcome).toMatchObject({ status: 'ok', threshold: 0.5 });
    expect(outcome.results.map((entry) => entry.kept)).toEqual([
      true,
      true,
      false,
    ]);
  });

  it.each([
    ['missing', undefined],
    ['all zero', { ...emptyUsage }],
    ['contradicting', { ...emptyUsage, calls: 7 }],
  ])(
    'sums the chain usage from the step attempts when its own total is %s',
    async (_label, total) => {
      const spent = { ...emptyUsage, calls: 1, inputTokens: 500 };
      const outcome = await withFallback([
        {
          reranker: scripted(
            'cross-encoder',
            () =>
              ({
                results: [
                  { index: 1, score: 1 },
                  { index: 0, score: 0 },
                ],
                provider: 'cross-encoder',
                model: 'm',
                status: 'ok',
                reason: null,
                attempts: [
                  {
                    provider: 'cross-encoder',
                    model: 'm',
                    status: 'ok',
                    reason: null,
                    durationMs: 1,
                    usage: spent,
                  },
                ],
                durationMs: 1,
                usage: total,
              }) as unknown as RerankOutcome
          ),
        },
      ]).rank({ query: 'q', documents: ['a', 'b'] });
      expect(outcome.status).toBe('ok');
      expect(outcome.attempts.map((attempt) => attempt.usage)).toEqual([spent]);
      expect(outcome.usage).toEqual(spent);
    }
  );

  it('records a step that ran but reported no attempt, so the history keeps every step', async () => {
    expect(
      (
        await noneReranker({ reason: 'no_api_key' }).rank({
          query: 'q',
          documents: ['a'],
        })
      ).attempts
    ).toEqual([
      {
        provider: 'none',
        model: '',
        status: 'skipped',
        reason: 'no_api_key',
        durationMs: expect.any(Number),
        usage: emptyUsage,
      },
    ]);
    const silent = scripted('bm25', () => ({
      results: [
        { index: 1, score: 3 },
        { index: 0, score: 1 },
      ],
      provider: 'bm25',
      model: 'okapi',
      status: 'ok',
      reason: null,
      attempts: [],
      durationMs: 2,
      usage: { ...emptyUsage },
    }));
    const outcome = await withFallback([
      {
        reranker: scripted('cross-encoder', () => Promise.reject(new Error())),
      },
      { reranker: silent },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(
      outcome.attempts.map((attempt) => [
        attempt.provider,
        attempt.status,
        attempt.reason,
      ])
    ).toEqual([
      ['cross-encoder', 'failed', 'error'],
      ['bm25', 'ok', null],
    ]);
  });

  it('abandons a gate at the chain deadline and starts no step after it', async () => {
    const step = scripted('cross-encoder', (request) =>
      noneReranker().rank(request)
    );
    const hung = withFallback([{ reranker: step }], {
      gate: () => new Promise<null>(() => {}),
    });
    let startedAt = performance.now();
    const never = await within(
      1_000,
      hung.rank({ query: 'q', documents: ['a'], deadlineMs: 50 })
    );
    expect(never).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(performance.now() - startedAt).toBeLessThan(500);

    const slow = withFallback([{ reranker: step }], {
      gate: () =>
        new Promise<null>((resolve) => setTimeout(() => resolve(null), 200)),
    });
    startedAt = performance.now();
    expect(
      await slow.rank({ query: 'q', documents: ['a'], deadlineMs: 50 })
    ).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(performance.now() - startedAt).toBeLessThan(180);

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 20);
    startedAt = performance.now();
    expect(
      await slow.rank({
        query: 'q',
        documents: ['a'],
        signal: controller.signal,
      })
    ).toMatchObject({ status: 'skipped', reason: 'aborted' });
    expect(performance.now() - startedAt).toBeLessThan(180);
    expect(step.requests).toHaveLength(0);
  });

  it('a caller abort ends the wait for a step that ignores it, with or without a deadline', async () => {
    for (const deadlineMs of [undefined, 2_000]) {
      const hung = scripted('cross-encoder', () => new Promise(() => {}));
      const backup = scripted('bm25', () => {
        throw new Error('unreachable');
      });
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 20);
      const startedAt = performance.now();
      const outcome = await within(
        1_000,
        withFallback([
          {
            reranker: hung,
            ...(deadlineMs !== undefined ? { deadlineMs } : {}),
          },
          { reranker: backup },
        ]).rank({
          query: 'q',
          documents: ['a', 'b'],
          signal: controller.signal,
        })
      );
      expect(outcome).toMatchObject({ status: 'skipped', reason: 'aborted' });
      expect(performance.now() - startedAt).toBeLessThan(500);
      expect((outcome as RerankOutcome).attempts[0]).toMatchObject({
        provider: 'cross-encoder',
        status: 'failed',
        reason: 'aborted',
        usage: { calls: 1, estimatedTokens: 400, estimatedSearchUnits: 1 },
      });
      expect(backup.requests).toHaveLength(0);
    }
  });

  it('a cooperative step that honors the caller abort reports its own spend', async () => {
    const cooperative = scripted(
      'cross-encoder',
      (request) =>
        new Promise<RerankOutcome>((resolve) => {
          request.signal?.addEventListener('abort', () =>
            resolve({
              results: [
                { index: 0, score: null },
                { index: 1, score: null },
              ],
              provider: 'cross-encoder',
              model: 'm',
              status: 'skipped',
              reason: 'aborted',
              attempts: [],
              durationMs: 20,
              usage: { ...emptyUsage, calls: 1, inputTokens: 7 },
            })
          );
        })
    );
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 20);
    const outcome = await withFallback([{ reranker: cooperative }]).rank({
      query: 'q',
      documents: ['a', 'b'],
      signal: controller.signal,
    });
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'aborted',
      usage: { calls: 1, inputTokens: 7, estimatedTokens: 0 },
    });
  });

  it('accepts a full answer that arrives within the step grace after its deadline', async () => {
    const late = scripted(
      'cross-encoder',
      () =>
        new Promise<RerankOutcome>((resolve) =>
          setTimeout(
            () =>
              resolve({
                results: [
                  { index: 1, score: 0.9 },
                  { index: 0, score: 0.1 },
                ],
                provider: 'cross-encoder',
                model: 'm',
                status: 'ok',
                reason: null,
                attempts: [],
                durationMs: 120,
                usage: { ...emptyUsage, calls: 1 },
              }),
            120
          )
        )
    );
    const backup = scripted('bm25', () => {
      throw new Error('unreachable');
    });
    const outcome = await withFallback([
      { reranker: late, deadlineMs: 100 },
      { reranker: backup },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(outcome).toMatchObject({ status: 'ok', provider: 'cross-encoder' });
    expect(order(outcome)).toEqual([1, 0]);
    expect(backup.requests).toHaveLength(0);
  });

  it('reports empty input and a cancelled caller without calling a step', async () => {
    const step = scripted('bm25', () => {
      throw new Error('unreachable');
    });
    const chain = withFallback([{ reranker: step }]);
    expect(await chain.rank({ query: 'q', documents: [] })).toMatchObject({
      status: 'skipped',
      reason: 'empty',
      results: [],
    });
    const controller = new AbortController();
    controller.abort();
    expect(
      await chain.rank({
        query: 'q',
        documents: ['a'],
        signal: controller.signal,
      })
    ).toMatchObject({ status: 'skipped', reason: 'aborted' });
    expect(step.requests).toHaveLength(0);
    expect(() => withFallback([])).toThrow('invalid reranker chain');
    expect(() => withFallback([{ reranker: step, deadlineMs: 0.5 }])).toThrow(
      'invalid reranker chain'
    );
  });

  it('answers an invalid request with identity order over every document', async () => {
    const step = scripted('bm25', () => {
      throw new Error('unreachable');
    });
    const chain = withFallback([
      { reranker: step },
      { reranker: noneReranker() },
    ]);
    for (const request of [
      { query: '  ', documents: ['a', 'b'] },
      { query: 'q', documents: ['a', 'b'], deadlineMs: 0 },
    ]) {
      const outcome = await chain.rank(request);
      expect(outcome).toMatchObject({
        status: 'skipped',
        reason: 'invalid_request',
      });
      expect(outcome.results).toEqual([
        { index: 0, score: null },
        { index: 1, score: null },
      ]);
    }
    expect(step.requests).toHaveLength(0);
  });

  it('treats an attempt with a malformed usage counter or duration as a broken step', async () => {
    const answer = (attempt: Partial<RerankOutcome['attempts'][number]>) =>
      scripted('cross-encoder', () => ({
        results: [
          { index: 1, score: 0.9 },
          { index: 0, score: 0.1 },
        ],
        provider: 'cross-encoder',
        model: 'cross-encoder-model',
        status: 'ok',
        reason: null,
        attempts: [
          {
            provider: 'cross-encoder',
            model: 'cross-encoder-model',
            status: 'ok',
            reason: null,
            durationMs: 1,
            usage: { ...emptyUsage, calls: 1, inputTokens: 10 },
            ...attempt,
          },
        ],
        durationMs: 1,
        usage: { ...emptyUsage, calls: 1, inputTokens: 10 },
      }));
    for (const attempt of [
      { usage: { ...emptyUsage, calls: -3, inputTokens: Number.NaN } },
      { usage: {} as typeof emptyUsage },
      {
        usage: { ...emptyUsage, outputTokens: undefined as unknown as number },
      },
      { durationMs: -1 },
    ]) {
      const outcome = await withFallback([
        { reranker: answer(attempt) },
        { reranker: noneReranker() },
      ]).rank({ query: 'q', documents: ['a', 'b'] });
      expect(outcome).toMatchObject({ status: 'skipped', provider: 'none' });
      expect(outcome.attempts[0]).toMatchObject({
        provider: 'cross-encoder',
        status: 'failed',
      });
      for (const { usage, durationMs } of outcome.attempts) {
        expect(durationMs).toBeGreaterThanOrEqual(0);
        for (const value of Object.values(usage)) {
          expect(Number.isFinite(value) && value >= 0).toBe(true);
        }
        expect(Object.keys(usage).sort()).toEqual(
          Object.keys(emptyUsage).sort()
        );
      }
    }
    const healthy = await withFallback([
      { reranker: answer({}) },
      { reranker: noneReranker() },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(healthy).toMatchObject({ status: 'ok', provider: 'cross-encoder' });
  });

  it('treats a sparse attempts array as a broken step', async () => {
    const holes = scripted('bm25', () => ({
      results: [
        { index: 1, score: 3 },
        { index: 0, score: 1 },
      ],
      provider: 'bm25',
      model: 'okapi',
      status: 'ok',
      reason: null,
      attempts: new Array<RerankOutcome['attempts'][number]>(1),
      durationMs: 1,
      usage: { ...emptyUsage },
    }));
    const outcome = await withFallback([
      { reranker: holes },
      { reranker: noneReranker() },
    ]).rank({ query: 'q', documents: ['a', 'b'] });
    expect(outcome).toMatchObject({ status: 'skipped', provider: 'none' });
    expect(outcome.attempts).not.toContain(undefined);
    expect(outcome.attempts[0]).toMatchObject({ status: 'failed' });
  });

  it('refuses a sparse chain at construction', () => {
    expect(() => withFallback(new Array<FallbackStep>(2))).toThrow(
      'invalid reranker chain'
    );
  });

  it('ignores an async onAttempt that rejects, without an unhandled rejection', async () => {
    let outcome: RerankOutcome | undefined;
    const unhandled = await unhandledDuring(async () => {
      outcome = await withFallback(
        [
          { reranker: createBm25Reranker({ idf: () => 1, avgdl: 2 }) },
          { reranker: noneReranker() },
        ],
        {
          onAttempt: async () => {
            throw new Error('bookkeeping');
          },
        }
      ).rank({ query: 'alpha', documents: ['alpha beta', 'gamma'] });
    });
    expect(unhandled).toEqual([]);
    expect(outcome).toMatchObject({ status: 'ok', provider: 'bm25' });
  });

  it('bounds the whole chain by the request deadline, not the sum of its steps', async () => {
    const server = await stub((call) => {
      const reply = relevance(RELEVANCE)(call);
      return reply === 'hang' ? reply : { ...reply, delayMs: 600 };
    });
    const slow = (): OutcomeReranker =>
      createCrossEncoderReranker({
        url: `${server.url}/rerank`,
        model: 'qwen3-reranker-8b',
        apiKey: 'key',
      });
    const startedAt = performance.now();
    const outcome = await withFallback([
      { reranker: slow(), deadlineMs: 400 },
      { reranker: slow(), deadlineMs: 400 },
      { reranker: noneReranker() },
    ]).rank({ query: 'q', documents: DOCUMENTS, deadlineMs: 150 });
    const took = performance.now() - startedAt;
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(order(outcome)).toEqual([0, 1, 2, 3]);
    expect(server.calls).toHaveLength(1);
    expect(took).toBeLessThan(350);

    const unbounded = scripted('bm25', (request) =>
      noneReranker().rank(request)
    );
    await withFallback([{ reranker: unbounded }]).rank({
      query: 'q',
      documents: ['a'],
      deadlineMs: 100,
    });
    expect(unbounded.requests[0].deadlineMs).toBeLessThanOrEqual(100);
    expect(unbounded.requests[0].signal).toBeInstanceOf(AbortSignal);
  });
});

describe('sparse document arrays', () => {
  const sparse = (): string[] => new Array<string>(2);

  it('is invalid_request for every reranker, before any request', async () => {
    const server = await stub(relevance(RELEVANCE));
    const { model, sent } = fakeJev(() => 2);
    const rerankers: OutcomeReranker[] = [
      createSystemOneReranker({ model, rubric }),
      createSystemOneReranker({ model, rubric, mode: 'batch' }),
      createCrossEncoderReranker({
        url: `${server.url}/rerank`,
        model: 'm',
        apiKey: 'k',
      }),
      createBm25Reranker({ idf: () => 1, avgdl: 2 }),
      withFallback([
        { reranker: createBm25Reranker({ idf: () => 1, avgdl: 2 }) },
        { reranker: noneReranker() },
      ]),
      noneReranker(),
    ];
    for (const reranker of rerankers) {
      const outcome = await reranker.rank({ query: 'q', documents: sparse() });
      expect(outcome).toMatchObject({
        status: 'skipped',
        reason: 'invalid_request',
        usage: emptyUsage,
      });
      expect(outcome.results).toEqual([
        { index: 0, score: null },
        { index: 1, score: null },
      ]);
    }
    expect(sent).toHaveLength(0);
    expect(server.calls).toHaveLength(0);
  });

  it('refuses a sparse rubric at construction', () => {
    const { model } = fakeJev(() => 2);
    expect(() =>
      createSystemOneReranker({ model, rubric: new Array<string>(3) })
    ).toThrow('invalid reranker configuration');
  });
});

describe('malformed request signals', () => {
  const malformed: Array<[string, unknown]> = [
    ['an empty object', {}],
    ['null', null],
    ['a string', 'nope'],
    [
      'a duck-typed signal',
      {
        aborted: false,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
      },
    ],
  ];

  it.each(malformed)(
    'withFallback answers %s as invalid_request, with or without a deadline or gate, and never rejects',
    async (_label, signal) => {
      const step = scripted('cross-encoder', () => {
        throw new Error('never reached');
      });
      const shapes: Array<[FallbackStep[], { deadlineMs?: number }, boolean]> =
        [
          [[{ reranker: step }], {}, false],
          [[{ reranker: step }], { deadlineMs: 500 }, false],
          [[{ reranker: step, deadlineMs: 500 }], {}, false],
          [[{ reranker: step }], {}, true],
        ];
      for (const [steps, extra, gated] of shapes) {
        const gate = jest.fn(() => null);
        const outcome = await withFallback(steps, gated ? { gate } : {}).rank({
          query: 'q',
          documents: ['a', 'b'],
          signal: signal as AbortSignal,
          ...extra,
        });
        expect(outcome).toMatchObject({
          status: 'skipped',
          reason: 'invalid_request',
          attempts: [],
          usage: emptyUsage,
        });
        expect(gate).not.toHaveBeenCalled();
      }
      expect(step.requests).toHaveLength(0);
    }
  );

  it.each(malformed)(
    'every reranker and probePin answer %s as invalid_request before any request',
    async (_label, signal) => {
      const server = await stub(relevance(RELEVANCE));
      const { model, sent } = fakeJev(() => 2);
      const rejecting = jest.fn(() =>
        Promise.resolve({ results: [], model: 'm', usage: null })
      );
      const rerankers: OutcomeReranker[] = [
        createSystemOneReranker({ model, rubric }),
        createSystemOneReranker({ model, rubric, mode: 'batch' }),
        createCrossEncoderReranker({
          url: `${server.url}/rerank`,
          model: 'm',
          apiKey: 'k',
        }),
        createBm25Reranker({ idf: () => 1, avgdl: 2 }),
        toOutcomeReranker(
          { id: 'rejecting', rerank: rejecting },
          { kind: 'cross-encoder', model: 'm' }
        ),
        noneReranker(),
      ];
      for (const reranker of rerankers) {
        const outcome = await reranker.rank({
          query: 'q',
          documents: ['a', 'b'],
          signal: signal as AbortSignal,
        });
        expect(outcome).toMatchObject({
          status: 'skipped',
          reason: 'invalid_request',
          usage: emptyUsage,
        });
      }
      const modelFor = jest.fn(() => model);
      await expect(
        probePin({
          modelFor,
          pin: 'jev-1.13.0',
          signal: signal as AbortSignal,
        })
      ).resolves.toEqual({ verified: false, reason: 'invalid_request' });
      expect(modelFor).not.toHaveBeenCalled();
      expect(rejecting).not.toHaveBeenCalled();
      expect(sent).toHaveLength(0);
      expect(server.calls).toHaveLength(0);
    }
  );

  it('accepts a real signal', async () => {
    const outcome = await withFallback([
      { reranker: createBm25Reranker({ idf: () => 1, avgdl: 2 }) },
    ]).rank({
      query: 'alpha',
      documents: ['gamma', 'alpha beta'],
      signal: new AbortController().signal,
      deadlineMs: 500,
    });
    expect(outcome).toMatchObject({ status: 'ok', reason: null });
  });

  it('withFallback resolves even when reading the request throws mid-chain', async () => {
    let reads = 0;
    const request = {
      query: 'q',
      documents: ['a', 'b'],
      get signal(): AbortSignal | undefined {
        reads += 1;
        if (reads > 1) {
          throw new Error('signal getter');
        }
        return undefined;
      },
    };
    const outcome = await withFallback([{ reranker: noneReranker() }]).rank(
      request
    );
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'error',
      attempts: [],
      usage: emptyUsage,
    });
  });
});

describe('probePin', () => {
  /** A host that 404s unknown models and echoes `serves` for the rest. */
  function host(known: string[], echo?: (model: string) => string) {
    return stub((call) => {
      const model = String(call.body.model);
      if (!known.includes(model)) {
        return { status: 404 };
      }
      return {
        body: {
          model: echo ? echo(model) : model,
          answers: { ok: { type: 'noul', noul: 0.9 } },
        },
      };
    });
  }

  it('verifies a pin only when the sentinel is refused with 404 and the pin echoes itself', async () => {
    const server = await host(['jev-1.13.0']);
    const modelFor = (model: string) => jev(server.url, { model });
    expect(await probePin({ modelFor, pin: 'jev-1.13.0' })).toEqual({
      verified: true,
      echoed: 'jev-1.13.0',
      sentinelStatus: 404,
    });
    expect(server.calls.map((call) => call.body.model)).toEqual([
      'jev-0.0.0-pin-sentinel',
      'jev-1.13.0',
    ]);
  });

  it('a host that answers the sentinel proves nothing', async () => {
    const server = await stub((call) => ({
      body: {
        model: call.body.model,
        answers: { ok: { type: 'noul', noul: 0.9 } },
      },
    }));
    expect(
      await probePin({
        modelFor: (model) => jev(server.url, { model }),
        pin: 'jev-1.13.0',
      })
    ).toEqual({
      verified: false,
      reason: 'sentinel_accepted',
      echoed: 'jev-0.0.0-pin-sentinel',
      sentinelStatus: 200,
    });
    expect(server.calls).toHaveLength(1);
  });

  it('reports the success status the sentinel really got, and none from a model that reports no status', async () => {
    const server = await stub((call) => ({
      status: 201,
      body: {
        model: call.body.model,
        answers: { ok: { type: 'noul', noul: 0.9 } },
      },
    }));
    expect(
      await probePin({
        modelFor: (model) => jev(server.url, { model }),
        pin: 'jev-1.13.0',
      })
    ).toEqual({
      verified: false,
      reason: 'sentinel_accepted',
      echoed: 'jev-0.0.0-pin-sentinel',
      sentinelStatus: 201,
    });
    const injected = (model: string): DecisionModel => ({
      id: 'injected',
      model,
      decide: async () => ({ model, answers: {}, usage: null }),
    });
    expect(await probePin({ modelFor: injected, pin: 'jev-1.13.0' })).toEqual({
      verified: false,
      reason: 'sentinel_accepted',
      echoed: 'jev-0.0.0-pin-sentinel',
    });
  });

  it('a pin answered as another model is echo_mismatch', async () => {
    const server = await host(['jev-1.13.0'], () => 'jev-1.14.0');
    expect(
      await probePin({
        modelFor: (model) => jev(server.url, { model }),
        pin: 'jev-1.13.0',
      })
    ).toEqual({
      verified: false,
      reason: 'echo_mismatch',
      echoed: 'jev-1.14.0',
      sentinelStatus: 404,
    });
  });

  it('a refused credential is its own failure, not a verdict on the pin, and never throws', async () => {
    const server = await stub(() => ({ status: 401 }));
    expect(
      await probePin({
        modelFor: (model) => jev(server.url, { model }, 'key'),
        pin: 'jev-1.13.0',
      })
    ).toEqual({ verified: false, reason: 'unauthorized', sentinelStatus: 401 });
    expect(
      await probePin({
        modelFor: () => {
          throw new Error('no host');
        },
        pin: 'jev-1.13.0',
      })
    ).toEqual({ verified: false, reason: 'error' });
  });

  it('reports no sentinel status when an open breaker refused the sentinel without a request', async () => {
    const server = await stub(() => ({ status: 402 }));
    const breaker = createBreaker();
    const modelFor = (model: string) =>
      withBreaker(jev(server.url, { model }, 'key'), breaker);
    expect(await probePin({ modelFor, pin: 'jev-1.13.0' })).toEqual({
      verified: false,
      reason: 'quota',
      sentinelStatus: 402,
    });
    expect(breaker.open).toBe(true);
    expect(await probePin({ modelFor, pin: 'jev-1.13.0' })).toEqual({
      verified: false,
      reason: 'circuit_open',
    });
    expect(server.calls).toHaveLength(1);
  });
});

describe('probePin input and deadlines', () => {
  /** A routing front that 404s a missing or blank model, in front of a backend that aliases every named one. */
  async function aliasing(): Promise<Stub> {
    return stub((call) => {
      const model = call.body.model;
      if (typeof model !== 'string' || model.trim() === '') {
        return { status: 404 };
      }
      return {
        body: { model, answers: { ok: { type: 'noul', noul: 0.9 } } },
      };
    });
  }

  it.each([[''], ['   ']])(
    'refuses a blank sentinel %j without a request',
    async (sentinel) => {
      const server = await aliasing();
      expect(
        await probePin({
          modelFor: (model) => jev(server.url, { model }),
          pin: 'jev-1.13.0',
          sentinel,
        })
      ).toEqual({ verified: false, reason: 'invalid_request' });
      expect(server.calls).toHaveLength(0);
    }
  );

  it('refuses a sentinel that is the pin once both are trimmed', async () => {
    const server = await aliasing();
    expect(
      await probePin({
        modelFor: (model) => jev(server.url, { model }),
        pin: 'jev-1.13.0',
        sentinel: ' jev-1.13.0 ',
      })
    ).toEqual({ verified: false, reason: 'invalid_request' });
    expect(server.calls).toHaveLength(0);
  });

  it('still catches an aliasing host with a custom sentinel', async () => {
    const server = await aliasing();
    expect(
      await probePin({
        modelFor: (model) => jev(server.url, { model }),
        pin: 'jev-1.13.0',
        sentinel: ' jev-9.9.9-missing ',
      })
    ).toMatchObject({ verified: false, reason: 'sentinel_accepted' });
    expect(server.calls.map((call) => call.body.model)).toEqual([
      'jev-9.9.9-missing',
    ]);
  });

  describe('a model that ignores its deadline and signal', () => {
    const modelFor = (model: string): DecisionModel => ({
      id: 'hanging',
      model,
      decide: () => new Promise(() => {}),
    });

    it('is bounded at timeoutMs', async () => {
      await expect(
        within(1_000, probePin({ modelFor, pin: 'jev-1.13.0', timeoutMs: 50 }))
      ).resolves.toEqual({ verified: false, reason: 'timeout' });
    });

    it('is bounded by the caller abort mid-flight', async () => {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 50);
      await expect(
        within(
          1_000,
          probePin({ modelFor, pin: 'jev-1.13.0', signal: controller.signal })
        )
      ).resolves.toEqual({ verified: false, reason: 'aborted' });
    });

    it('answers a signal that already aborted at once', async () => {
      await expect(
        within(
          500,
          probePin({ modelFor, pin: 'jev-1.13.0', signal: AbortSignal.abort() })
        )
      ).resolves.toEqual({ verified: false, reason: 'aborted' });
    });
  });
});

describe('toOutcomeReranker with a reranker that ignores its deadline and signal', () => {
  const hanging = toOutcomeReranker(
    { id: 'hanging', rerank: () => new Promise(() => {}) },
    { kind: 'cross-encoder', model: 'm' }
  );

  it('settles at deadlineMs as a charged timeout', async () => {
    await expect(
      within(
        1_000,
        hanging.rank({ query: 'q', documents: ['a', 'b'], deadlineMs: 50 })
      )
    ).resolves.toMatchObject({
      status: 'skipped',
      reason: 'timeout',
      usage: { calls: 1, estimatedTokens: 400, estimatedSearchUnits: 1 },
    });
  });

  it('settles when the caller aborts mid-flight', async () => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 50);
    await expect(
      within(
        1_000,
        hanging.rank({
          query: 'q',
          documents: ['a'],
          signal: controller.signal,
        })
      )
    ).resolves.toMatchObject({ status: 'skipped', reason: 'aborted' });
  });
});

describe('toOutcomeReranker deadline', () => {
  /** A reranker that gives up at its own `timeoutMs` (200 ms unless overridden) on a host that answers after 1 s. */
  const configured = (): {
    reranker: OutcomeReranker;
    seen: Array<number | undefined>;
  } => {
    const seen: Array<number | undefined> = [];
    const reranker = toOutcomeReranker(
      {
        id: 'configured',
        rerank: (request) => {
          seen.push(request.timeoutMs);
          return new Promise((resolve, reject) => {
            const answer = setTimeout(
              () =>
                resolve({
                  results: [
                    { index: 1, score: 2 },
                    { index: 0, score: 1 },
                  ],
                  model: 'm',
                  usage: null,
                }),
              1_000
            );
            setTimeout(() => {
              clearTimeout(answer);
              reject(new DecisionError('timeout', 'gave up'));
            }, request.timeoutMs ?? 200);
          });
        },
      },
      { kind: 'cross-encoder', model: 'm' }
    );
    return { reranker, seen };
  };

  it('caps the wrapped reranker at its own shorter timeout instead of extending it to the deadline', async () => {
    const { reranker, seen } = configured();
    const startedAt = performance.now();
    const outcome = await reranker.rank({
      query: 'q',
      documents: ['a', 'b'],
      deadlineMs: 1_500,
    });
    expect(performance.now() - startedAt).toBeLessThan(800);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(seen).toEqual([undefined]);
  });

  it('lets a chain fall back when the wrapped step times out on its own', async () => {
    const { reranker } = configured();
    const startedAt = performance.now();
    const outcome = await withFallback([
      { reranker },
      { reranker: createBm25Reranker({ idf: () => 1, avgdl: 2 }) },
    ]).rank({ query: 'beta', documents: ['alpha', 'beta'], deadlineMs: 1_500 });
    expect(performance.now() - startedAt).toBeLessThan(800);
    expect(outcome).toMatchObject({
      status: 'degraded',
      reason: 'timeout',
      provider: 'bm25',
    });
  });
});

describe('noneReranker requests', () => {
  const malformed: Array<[string, unknown]> = [
    ['a blank query', { query: ' ', documents: ['a'] }],
    ['sparse documents', { query: 'q', documents: new Array<string>(2) }],
    ['a non-string document', { query: 'q', documents: ['a', 1] }],
    ['a negative deadline', { query: 'q', documents: ['a'], deadlineMs: -5 }],
    ['an endless deadline', { query: 'q', documents: ['a'], deadlineMs: 1e12 }],
    [
      'a duck-typed signal',
      { query: 'q', documents: ['a'], signal: { aborted: false } },
    ],
    ['no request', null],
  ];

  it.each(malformed)('is invalid_request for %s', async (_label, request) => {
    const outcome = await noneReranker({ reason: 'budget' }).rank(
      request as RankRequest
    );
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'invalid_request',
      usage: emptyUsage,
    });
    expect(outcome.attempts).toEqual([
      expect.objectContaining({ status: 'skipped', reason: 'invalid_request' }),
    ]);
  });

  it('is empty without documents, and its own reason otherwise', async () => {
    await expect(
      noneReranker().rank({ query: 'q', documents: [] })
    ).resolves.toMatchObject({ status: 'skipped', reason: 'empty' });
    await expect(
      noneReranker().rank({ query: 'q', documents: ['a'] })
    ).resolves.toMatchObject({ status: 'skipped', reason: 'disabled' });
    await expect(
      noneReranker({ reason: 'budget' }).rank({ query: 'q', documents: ['a'] })
    ).resolves.toMatchObject({ status: 'skipped', reason: 'budget' });
  });
});

describe('withFallback steps that reject once stopped', () => {
  /** Honors its signal by rejecting, as a reranker that breaks the never-rejects contract might. */
  const rejectsOnAbort = (): OutcomeReranker =>
    scripted(
      'cross-encoder',
      (request) =>
        new Promise<RerankOutcome>((_resolve, reject) => {
          request.signal?.addEventListener('abort', () =>
            reject(new Error('stopped'))
          );
        })
    );
  /** Honors its signal by resolving nothing. */
  const resolvesNothingOnAbort = (): OutcomeReranker =>
    scripted(
      'cross-encoder',
      (request) =>
        new Promise<RerankOutcome>((resolve) => {
          request.signal?.addEventListener('abort', () =>
            resolve(undefined as unknown as RerankOutcome)
          );
        })
    );
  const bm25 = (): OutcomeReranker =>
    createBm25Reranker({ idf: () => 1, avgdl: 2 });

  it.each([
    ['rejects', rejectsOnAbort],
    ['resolves nothing', resolvesNothingOnAbort],
  ])('is a timeout when a step %s at its deadline', async (_label, step) => {
    const outcome = await withFallback([
      { reranker: step(), deadlineMs: 30 },
      { reranker: bm25() },
    ]).rank({ query: 'beta', documents: ['alpha', 'beta'] });
    expect(outcome).toMatchObject({ status: 'degraded', reason: 'timeout' });
    expect(outcome.attempts[0]).toMatchObject({
      status: 'failed',
      reason: 'timeout',
    });
  });

  it.each([
    ['rejects', rejectsOnAbort],
    ['resolves nothing', resolvesNothingOnAbort],
  ])('is aborted when a step %s on the caller abort', async (_label, step) => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 30);
    const outcome = await withFallback([
      { reranker: step() },
      { reranker: bm25() },
    ]).rank({
      query: 'beta',
      documents: ['alpha', 'beta'],
      signal: controller.signal,
    });
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'aborted' });
    expect(outcome.attempts[0]).toMatchObject({
      status: 'failed',
      reason: 'aborted',
    });
  });

  it('is still an error when a step rejects on its own', async () => {
    const outcome = await withFallback([
      {
        reranker: scripted('cross-encoder', () =>
          Promise.reject(new Error('boom'))
        ),
        deadlineMs: 500,
      },
      { reranker: bm25() },
    ]).rank({ query: 'beta', documents: ['alpha', 'beta'] });
    expect(outcome).toMatchObject({ status: 'degraded', reason: 'error' });
  });
});

describe('toOutcomeReranker and rerankCost', () => {
  it('turns a rejecting Reranker into reasons and validates its order', async () => {
    const failing = toOutcomeReranker(
      {
        id: 'host',
        rerank: () =>
          Promise.reject(new Error('provider said something private')),
      },
      { kind: 'cross-encoder', model: 'host-model' }
    );
    const failed = await failing.rank({ query: 'q', documents: ['a'] });
    expect(failed).toMatchObject({ status: 'skipped', reason: 'error' });
    expect(JSON.stringify(failed)).not.toContain('private');
    const unordered = toOutcomeReranker(
      {
        id: 'host',
        rerank: async () => ({
          results: [
            { index: 0, score: 0.1 },
            { index: 1, score: 0.9 },
          ],
          model: 'host-model',
          usage: null,
        }),
      },
      { kind: 'cross-encoder' }
    );
    expect(
      (await unordered.rank({ query: 'q', documents: ['a', 'b'] })).reason
    ).toBe('invalid_results');
  });

  it('prices every attempt from a caller-supplied table and lists what it cannot price', () => {
    const attempt = (
      provider: RerankOutcome['provider'],
      model: string,
      usage: Partial<typeof emptyUsage>
    ): RerankOutcome['attempts'][number] => ({
      provider,
      model,
      status: 'failed',
      reason: 'timeout',
      durationMs: 1,
      usage: { ...emptyUsage, ...usage },
    });
    const cost = rerankCost(
      {
        attempts: [
          attempt('systemone', 'jev-1.13.0', {
            inputTokens: 1_000_000,
            estimatedTokens: 1_000_000,
          }),
          attempt('cross-encoder', 'cohere-rerank-v3-5', {
            searchUnits: 1,
            estimatedSearchUnits: 1,
          }),
          attempt('cross-encoder', 'unlisted', { inputTokens: 10 }),
          attempt('bm25', 'okapi', {}),
        ],
      },
      {
        systemone: { 'jev-*': { inputPerMTok: 0.042 } },
        'cross-encoder': { 'cohere-rerank-v3-5': { perSearchUnit: 0.002 } },
      }
    );
    expect(cost).toEqual({
      usd: 0.088,
      unpriced: [{ provider: 'cross-encoder', model: 'unlisted' }],
    });
  });
});

describe('rerankCost rates', () => {
  const attempt = (
    provider: RerankOutcome['provider'],
    model: string,
    usage: Partial<typeof emptyUsage>
  ): RerankOutcome['attempts'][number] => ({
    provider,
    model,
    status: 'ok',
    reason: null,
    durationMs: 1,
    usage: { ...emptyUsage, ...usage },
  });

  it('lists an attempt as unpriced when its entry has no rate for a reported dimension it used', () => {
    expect(
      rerankCost(
        { attempts: [attempt('cross-encoder', 'jina', { inputTokens: 5000 })] },
        { 'cross-encoder': { '*': { perSearchUnit: 0.002 } } }
      )
    ).toEqual({
      usd: 0,
      unpriced: [{ provider: 'cross-encoder', model: 'jina' }],
    });
    expect(
      rerankCost(
        {
          attempts: [
            attempt('cross-encoder', 'm', {
              inputTokens: 1_000_000,
              searchUnits: 10,
            }),
          ],
        },
        { 'cross-encoder': { m: { inputPerMTok: 0.05 } } }
      )
    ).toEqual({
      usd: 0,
      unpriced: [{ provider: 'cross-encoder', model: 'm' }],
    });
    expect(
      rerankCost(
        { attempts: [attempt('systemone', 'jev', { inputTokens: 1_000_000 })] },
        { systemone: { jev: {} } }
      )
    ).toEqual({ usd: 0, unpriced: [{ provider: 'systemone', model: 'jev' }] });
  });

  it('treats an invalid rate as missing, and an explicit 0 as free', () => {
    for (const inputPerMTok of [Number.NaN, -1, '0.05' as unknown as number]) {
      expect(
        rerankCost(
          { attempts: [attempt('systemone', 'jev', { inputTokens: 10 })] },
          { systemone: { jev: { inputPerMTok } } }
        )
      ).toEqual({
        usd: 0,
        unpriced: [{ provider: 'systemone', model: 'jev' }],
      });
    }
    expect(
      rerankCost(
        {
          attempts: [
            attempt('cross-encoder', 'm', {
              inputTokens: 1_000_000,
              searchUnits: 10,
            }),
          ],
        },
        { 'cross-encoder': { m: { inputPerMTok: 0, perSearchUnit: 0.002 } } }
      )
    ).toEqual({ usd: 0.02, unpriced: [] });
  });

  it('prices an estimate by whichever of its rates the entry has, and lists it only when it has neither', () => {
    const abandoned = attempt('cross-encoder', 'm', {
      estimatedTokens: 800,
      estimatedSearchUnits: 1,
    });
    expect(
      rerankCost(
        { attempts: [abandoned] },
        { 'cross-encoder': { m: { perSearchUnit: 0.002 } } }
      )
    ).toEqual({ usd: 0.002, unpriced: [] });
    expect(
      rerankCost(
        { attempts: [abandoned] },
        { 'cross-encoder': { m: { inputPerMTok: 1 } } }
      )
    ).toEqual({ usd: 0.0008, unpriced: [] });
    expect(
      rerankCost(
        { attempts: [abandoned] },
        { 'cross-encoder': { m: { outputPerMTok: 1 } } }
      )
    ).toEqual({
      usd: 0,
      unpriced: [{ provider: 'cross-encoder', model: 'm' }],
    });
  });
});

describe('a request deadline caps a reranker’s own timeout', () => {
  /** A Jev model whose requests hang until their abort. */
  const hungJev = (): DecisionModel => ({
    id: 'hung-jev',
    model: 'jev-1.13.0',
    decide: (request) =>
      new Promise((_, reject) => {
        request.signal?.addEventListener('abort', () =>
          reject(new DecisionError('aborted', 'aborted', { provider: 'x' }))
        );
      }),
  });
  const bm25 = (): OutcomeReranker =>
    createBm25Reranker({ idf: new Map([['alpha', 1]]), avgdl: 1 });

  it.each(['document', 'batch'] as const)(
    'a %s-mode System One step with a 200 ms timeout falls back to BM25 at 200 ms under a 1500 ms request deadline',
    async (mode) => {
      const chain = withFallback([
        {
          reranker: createSystemOneReranker({
            model: hungJev(),
            rubric,
            timeoutMs: 200,
            mode,
          }),
        },
        { reranker: bm25() },
      ]);
      const startedAt = performance.now();
      const outcome = await chain.rank({
        query: 'alpha',
        documents: ['beta', 'alpha'],
        deadlineMs: 1_500,
      });
      expect(performance.now() - startedAt).toBeLessThan(900);
      expect(outcome).toMatchObject({
        status: 'degraded',
        provider: 'bm25',
        reason: 'timeout',
      });
      expect(outcome.attempts.map((attempt) => attempt.provider)).toEqual([
        'systemone',
        'bm25',
      ]);
    }
  );

  it('a cross-encoder with a 200 ms timeout gives up at 200 ms under a 1500 ms request deadline', async () => {
    const server = await stub(() => 'hang');
    const startedAt = performance.now();
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'ce',
      requiresAuth: false,
      timeoutMs: 200,
    }).rank({ query: 'q', documents: ['a'], deadlineMs: 1_500 });
    expect(performance.now() - startedAt).toBeLessThan(900);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
  });

  it('a shorter request deadline still shortens the reranker’s timeout', async () => {
    const startedAt = performance.now();
    const outcome = await createSystemOneReranker({
      model: hungJev(),
      rubric,
      timeoutMs: 5_000,
    }).rank({ query: 'q', documents: ['a'], deadlineMs: 100 });
    expect(performance.now() - startedAt).toBeLessThan(900);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
  });
});

describe('a model that reports its requests is counted at the transport', () => {
  /** An HTTP Jev model whose credential is `apiKey`, against a server that counts what reached it. */
  const httpJev = (
    url: string,
    apiKey: DecisionCredential | undefined,
    maxRetries = 0
  ): DecisionModel =>
    createHttpDecisionModel({
      providerId: 'jev',
      endpoint: `${url}/v1/systemone`,
      model: 'jev-1.13.0',
      dialect: 'systemone',
      requiresAuth: apiKey !== undefined,
      apiKey,
      maxRetries,
    });
  const docs = ['alpha', 'beta', 'gamma'];

  it('the HTTP model reports its requests, and a breaker passes that on', () => {
    const model = httpJev('http://127.0.0.1:1', undefined);
    expect(model.reportsRequests).toBe(true);
    expect(withBreaker(model).reportsRequests).toBe(true);
    expect(
      withBreaker({ ...fakeJev(() => 1).model }).reportsRequests
    ).toBeUndefined();
  });

  it('a minter that returns no token sends nothing: no call, a skipped attempt', async () => {
    const server = await stub(scored(SCORES));
    for (const model of [
      httpJev(server.url, async () => ''),
      withBreaker(httpJev(server.url, async () => '')),
    ]) {
      const outcome = await createSystemOneReranker({ model, rubric }).rank({
        query: 'q',
        documents: docs,
      });
      expect(outcome).toMatchObject({
        status: 'skipped',
        reason: 'unauthorized',
      });
      expect(outcome.usage).toEqual(emptyUsage);
      expect(outcome.attempts[0].status).toBe('skipped');
    }
    expect(server.calls).toHaveLength(0);
  });

  it('a minter still pending at the deadline charges no call and no estimate', async () => {
    const server = await stub(scored(SCORES));
    const outcome = await createSystemOneReranker({
      model: httpJev(server.url, () => new Promise<string>(() => {})),
      rubric,
    }).rank({ query: 'q', documents: docs, deadlineMs: 150 });
    expect(server.calls).toHaveLength(0);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(outcome.usage).toEqual(emptyUsage);
    expect(outcome.attempts[0].status).toBe('skipped');
    expect(
      rerankCost(outcome, { systemone: { '*': { inputPerMTok: 1 } } }).usd
    ).toBe(0);
  });

  it('a state that cannot be serialized sends nothing, on the first pass or the re-issue', async () => {
    const server = await stub(scored(SCORES));
    const outcome = await createSystemOneReranker({
      model: httpJev(server.url, undefined),
      rubric,
      state: (query, document) =>
        ({ query, document, n: BigInt(1) }) as unknown as Record<
          string,
          string
        >,
    }).rank({ query: 'q', documents: docs });
    expect(server.calls).toHaveLength(0);
    expect(outcome.usage).toEqual(emptyUsage);
    expect(outcome.attempts[0].status).toBe('skipped');
  });

  it('batch mode with a minter still pending at the deadline charges no call and no estimate', async () => {
    const server = await stub(scored(SCORES));
    const outcome = await createSystemOneReranker({
      model: httpJev(server.url, () => new Promise<string>(() => {})),
      rubric,
      mode: 'batch',
    }).rank({ query: 'q', documents: docs, deadlineMs: 150 });
    expect(server.calls).toHaveLength(0);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(outcome.usage).toEqual(emptyUsage);
    expect(outcome.attempts[0].status).toBe('skipped');
  });

  it.each(['document', 'batch'] as const)(
    'a caller abort that lands as a %s-mode request is scheduled sends, counts, and charges nothing',
    async (mode) => {
      const server = await stub(scored(SCORES));
      const controller = new AbortController();
      const pending = createSystemOneReranker({
        model: httpJev(server.url, undefined),
        rubric,
        mode,
      }).rank({
        query: 'q',
        documents: docs,
        signal: controller.signal,
        deadlineMs: 1_000,
      });
      queueMicrotask(() => controller.abort());
      const outcome = await pending;
      expect(server.calls).toHaveLength(0);
      expect(outcome).toMatchObject({ status: 'skipped', reason: 'aborted' });
      expect(outcome.usage).toEqual(emptyUsage);
    }
  );

  it('a model that does not report requests is not dispatched once an abort lands before dispatch', async () => {
    const { model, sent } = fakeJev(() => 1);
    const controller = new AbortController();
    const pending = createSystemOneReranker({ model, rubric }).rank({
      query: 'q',
      documents: docs,
      signal: controller.signal,
      deadlineMs: 1_000,
    });
    queueMicrotask(() => controller.abort());
    const outcome = await pending;
    expect(sent).toHaveLength(0);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'aborted' });
    expect(outcome.usage).toEqual(emptyUsage);
    expect(outcome.attempts[0].status).toBe('skipped');
  });
});

describe('a deadline during backoff charges no estimate and reports the reply', () => {
  const throttled = (): Reply => ({
    status: 429,
    headers: { 'retry-after': '5' },
  });

  it('cross-encoder: a 429 with Retry-After is rate_limited, not an abandoned request', async () => {
    const server = await stub(throttled);
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'ce',
      requiresAuth: false,
      maxRetries: 2,
    }).rank({ query: 'q', documents: ['a', 'b', 'c'], deadlineMs: 300 });
    expect(server.calls).toHaveLength(1);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'rate_limited',
    });
    expect(outcome.usage).toMatchObject({
      calls: 1,
      estimatedTokens: 0,
      estimatedSearchUnits: 0,
    });
    expect(
      rerankCost(outcome, {
        'cross-encoder': { '*': { perSearchUnit: 0.002 } },
      }).usd
    ).toBe(0);
  });

  it('cross-encoder: a 500 answered before the deadline is not estimated', async () => {
    const server = await stub(() => ({ status: 500 }));
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'ce',
      requiresAuth: false,
      maxRetries: 1,
    }).rank({ query: 'q', documents: ['a', 'b', 'c'], deadlineMs: 150 });
    expect(server.calls).toHaveLength(1);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'error' });
    expect(outcome.usage).toMatchObject({
      calls: 1,
      estimatedTokens: 0,
      estimatedSearchUnits: 0,
    });
  });

  it.each(['document', 'batch'] as const)(
    'System One %s mode: a 429 with Retry-After is rate_limited and charges no estimate',
    async (mode) => {
      const server = await stub(throttled);
      const documents = ['alpha', 'beta', 'gamma'];
      const outcome = await createSystemOneReranker({
        model: jev(server.url, { maxRetries: 2 }),
        rubric,
        mode,
      }).rank({ query: 'q', documents, deadlineMs: 300 });
      const requests = mode === 'document' ? documents.length : 1;
      expect(server.calls).toHaveLength(requests);
      expect(outcome).toMatchObject({
        status: 'skipped',
        reason: 'rate_limited',
      });
      expect(outcome.usage).toMatchObject({
        calls: requests,
        estimatedTokens: 0,
      });
    }
  );

  it('System One: a request still in flight at the deadline is estimated', async () => {
    const server = await stub(() => 'hang');
    const outcome = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
    }).rank({ query: 'q', documents: ['alpha', 'beta'], deadlineMs: 150 });
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 1_200 });
  });

  it('System One: an always-429 host gets one request per document, with nothing re-issued', async () => {
    const server = await stub(() => ({ status: 429 }));
    const outcome = await createSystemOneReranker({
      model: jev(server.url),
      rubric,
    }).rank({ query: 'q', documents: ['alpha', 'beta', 'gamma'] });
    expect(server.calls).toHaveLength(3);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'rate_limited',
    });
    expect(outcome.usage.calls).toBe(3);
  });
});

describe('a deadline reports the backoff it interrupted, not an earlier one', () => {
  /** Answers each request with the next of `replies`, then hangs. */
  function sequence(replies: Reply[]): (call: Call, nth: number) => Reply {
    return (_, nth) => replies[nth] ?? 'hang';
  }

  it('cross-encoder: a 500 then a 429 whose Retry-After outlasts the deadline is rate_limited', async () => {
    const server = await stub(
      sequence([
        { status: 500 },
        { status: 429, headers: { 'retry-after': '5' } },
      ])
    );
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'ce',
      requiresAuth: false,
      maxRetries: 2,
    }).rank({ query: 'q', documents: ['a', 'b'], deadlineMs: 600 });
    expect(server.calls).toHaveLength(2);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'rate_limited',
    });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  it('cross-encoder: a 500 then a 401 whose refresh outlasts the deadline is a timeout', async () => {
    const server = await stub(sequence([{ status: 500 }, { status: 401 }]));
    const outcome = await createCrossEncoderReranker({
      url: `${server.url}/rerank`,
      model: 'ce',
      maxRetries: 2,
      apiKey: ({ refresh }) =>
        refresh ? new Promise<string>(() => {}) : Promise.resolve('k1'),
    }).rank({ query: 'q', documents: ['a', 'b'], deadlineMs: 600 });
    expect(server.calls).toHaveLength(2);
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  it('toOutcomeReranker: a 500 then a 429 backed off past the deadline is rate_limited', async () => {
    const outcome = await toOutcomeReranker(
      {
        id: 'reporting',
        reportsRequests: true,
        rerank: async (request) => {
          request.onRequest?.();
          request.onResponse?.(500);
          request.onRequest?.();
          request.onResponse?.(429, 5_000);
          return new Promise(() => {});
        },
      },
      { kind: 'cross-encoder', model: 'ce' }
    ).rank({ query: 'q', documents: ['a', 'b'], deadlineMs: 100 });
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'rate_limited',
    });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  /** A reporting model: `alpha` follows `alpha`'s script, every other document hangs before sending. */
  function reporting(
    alpha: (request: DecisionRequest) => Promise<void>
  ): DecisionModel {
    const score = {
      model: 'jev-1.13.0',
      answers: {
        relevance: {
          type: 'score' as const,
          score: 2,
          confidence: null,
          probabilities: null,
        },
      },
      usage: null,
    };
    return {
      id: 'reporting',
      model: 'jev-1.13.0',
      reportsRequests: true,
      decide: async (request) => {
        if ((request.state as { document: string }).document !== 'alpha') {
          return new Promise(() => {});
        }
        await alpha(request);
        return score;
      },
    };
  }

  it('System One: a backoff that ended in a score is not what the deadline interrupted', async () => {
    const model = reporting(async (request) => {
      request.onRequest?.();
      request.onResponse?.(500);
      await new Promise((resolve) => setTimeout(resolve, 10));
      request.onRequest?.();
      request.onResponse?.(200);
    });
    const outcome = await createSystemOneReranker({
      model,
      rubric,
      concurrency: 1,
    }).rank({ query: 'q', documents: ['alpha', 'beta'], deadlineMs: 150 });
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'timeout' });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  it('System One: a score on another document does not hide a backoff the deadline interrupted', async () => {
    const backingOff: DecisionModel = {
      id: 'reporting',
      model: 'jev-1.13.0',
      reportsRequests: true,
      decide: async (request) => {
        const document = (request.state as { document: string }).document;
        request.onRequest?.();
        if (document === 'alpha') {
          request.onResponse?.(503);
          return new Promise(() => {});
        }
        await new Promise((resolve) => setTimeout(resolve, 10));
        request.onResponse?.(200);
        return {
          model: 'jev-1.13.0',
          answers: {
            relevance: {
              type: 'score',
              score: 2,
              confidence: null,
              probabilities: null,
            },
          },
          usage: null,
        };
      },
    };
    const outcome = await createSystemOneReranker({
      model: backingOff,
      rubric,
    }).rank({ query: 'q', documents: ['alpha', 'beta'], deadlineMs: 150 });
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'error' });
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });
});

describe('a stopped System One step sends nothing more', () => {
  it('a 429 stops the step at once: the transport does not retry it, though the model allows retries', async () => {
    const server = await stub(() => ({ status: 429 }));
    const started = performance.now();
    const outcome = await createSystemOneReranker({
      model: jev(server.url, { maxRetries: 2 }),
      rubric,
    }).rank({ query: 'q', documents: ['alpha', 'beta', 'gamma'] });
    // The transport's two backoffs would take a second: none was waited out.
    expect(performance.now() - started).toBeLessThan(1_000);
    expect(server.calls).toHaveLength(3);
    expect(outcome).toMatchObject({
      status: 'skipped',
      reason: 'rate_limited',
    });
    expect(outcome.usage).toMatchObject({ calls: 3, estimatedTokens: 0 });
  });

  it('rejects rerank() with the 429 and its Retry-After instead of waiting it out', async () => {
    const server = await stub(() => ({
      status: 429,
      headers: { 'retry-after': '5' },
    }));
    const started = performance.now();
    const error = await createSystemOneReranker({
      model: jev(server.url, { maxRetries: 2 }),
      rubric,
      concurrency: 1,
    })
      .rerank({ query: 'q', documents: ['alpha', 'beta', 'gamma'] })
      .then(
        () => undefined,
        (thrown: unknown) => thrown
      );
    expect(performance.now() - started).toBeLessThan(1_000);
    expect(server.calls).toHaveLength(1);
    expect(error).toBeInstanceOf(DecisionError);
    expect(error).toMatchObject({
      failure: 'rate_limited',
      status: 429,
      retryAfterMs: 5_000,
    });
  });

  it('halts a document backing off from a 5xx once another document stops the step', async () => {
    const server = await stub((call) =>
      documentOf(call) === 'alpha'
        ? { status: 503 }
        : { status: 402, delayMs: 50 }
    );
    const outcome = await createSystemOneReranker({
      model: jev(server.url, { maxRetries: 2 }),
      rubric,
    }).rank({ query: 'q', documents: ['alpha', 'beta'] });
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'quota' });
    // The 503 would be retried 250 ms later; the 402 stopped the step first.
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(server.calls.map(documentOf).sort()).toEqual(['alpha', 'beta']);
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  it.each([
    ['after', 5, 50],
    ['before', 40, 5],
  ])(
    'halts a document whose 2xx body broke off when another document stops the step %s it',
    async (_, alphaMs, betaMs) => {
      const server = await stub((call) =>
        documentOf(call) === 'alpha'
          ? { truncate: true, delayMs: alphaMs }
          : { status: 402, delayMs: betaMs }
      );
      const started = performance.now();
      const outcome = await createSystemOneReranker({
        model: jev(server.url, { maxRetries: 2 }),
        rubric,
      }).rank({ query: 'q', documents: ['alpha', 'beta'] });
      expect(outcome).toMatchObject({ status: 'skipped', reason: 'quota' });
      // The broken body would be retried 250 ms later; the 402 stops the step.
      expect(performance.now() - started).toBeLessThan(250);
      await new Promise((resolve) => setTimeout(resolve, 400));
      expect(server.calls.map(documentOf).sort()).toEqual(['alpha', 'beta']);
      expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
    }
  );

  it('does not send a retry the model starts after a 2xx once the step has stopped', async () => {
    const sent: string[] = [];
    const model: DecisionModel = {
      id: 'retries-after-2xx',
      model: 'jev-1.13.0',
      reportsRequests: true,
      decide: async (request) => {
        const document = (request.state as { document: string }).document;
        const send = (): boolean => {
          request.onRequest?.();
          if (request.signal?.aborted === true) {
            return false;
          }
          sent.push(document);
          return true;
        };
        if (document === 'beta') {
          await new Promise((resolve) => setTimeout(resolve, 20));
          send();
          request.onResponse?.(402);
          throw new DecisionError('quota', 'quota', { status: 402 });
        }
        // A 2xx whose answer then fails, retried as the transport retries a network failure.
        send();
        request.onResponse?.(200);
        await new Promise((resolve) => setTimeout(resolve, 60));
        if (!send()) {
          throw new DecisionError('aborted', 'halted');
        }
        request.onResponse?.(200);
        throw new DecisionError('network', 'failed again');
      },
    };
    const outcome = await createSystemOneReranker({ model, rubric }).rank({
      query: 'q',
      documents: ['alpha', 'beta'],
    });
    expect(outcome).toMatchObject({ status: 'skipped', reason: 'quota' });
    expect(sent.sort()).toEqual(['alpha', 'beta']);
    expect(outcome.usage).toMatchObject({ calls: 2, estimatedTokens: 0 });
  });

  it('rejects rerank() with the error of the failure that stopped the step first', async () => {
    /** `alpha` answers as another checkpoint at once; `beta` is refused 20 ms later. */
    const model: DecisionModel = {
      id: 'mixed',
      model: 'jev-1.13.0',
      decide: async (request) => {
        if ((request.state as { document: string }).document !== 'alpha') {
          await new Promise((resolve) => setTimeout(resolve, 20));
          throw new DecisionError('unauthorized', 'refused', { status: 401 });
        }
        return {
          model: 'jev-0.9.0',
          answers: {
            relevance: {
              type: 'score',
              score: 1,
              confidence: null,
              probabilities: null,
            },
          },
          usage: null,
        };
      },
    };
    const reranker = createSystemOneReranker({
      model,
      rubric,
      pin: { model: 'jev-1.13.0' },
    });
    const outcome = await reranker.rank({
      query: 'q',
      documents: ['alpha', 'beta'],
    });
    expect(outcome.reason).toBe('pin_mismatch');
    await expect(
      reranker.rerank({ query: 'q', documents: ['alpha', 'beta'] })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
  });
});

describe('BM25 with a query it reads no terms in', () => {
  const stats = corpusStats(['café au lait', 'カフェ ラテ', 'coffee bean']);

  it.each(['カフェ', 'été', 'the and of'])(
    'skips %p as unsupported instead of an all-zero ranking',
    async (query) => {
      const outcome = await createBm25Reranker(stats).rank({
        query,
        documents: ['café au lait', 'カフェ ラテ'],
      });
      expect(outcome).toMatchObject({
        status: 'skipped',
        reason: 'unsupported',
      });
      expect(outcome.attempts[0].status).toBe('skipped');
    }
  );

  it('lets a chain move on to its next step', async () => {
    const failing = scripted('systemone', (request) =>
      noneReranker({ reason: 'timeout' }).rank(request)
    );
    const next = scripted('cross-encoder', (request) => ({
      results: request.documents.map((_, index) => ({
        index,
        score: request.documents.length - index,
      })),
      provider: 'cross-encoder',
      model: 'ce',
      status: 'ok',
      reason: null,
      attempts: [],
      durationMs: 0,
      usage: emptyUsage,
    }));
    const outcome = await withFallback([
      { reranker: failing },
      { reranker: createBm25Reranker(stats) },
      { reranker: next },
    ]).rank({ query: 'カフェ', documents: ['café au lait', 'カフェ ラテ'] });
    expect(outcome).toMatchObject({
      status: 'degraded',
      provider: 'cross-encoder',
    });
    expect(outcome.attempts.map((attempt) => attempt.provider)).toEqual([
      'none',
      'bm25',
      'cross-encoder',
    ]);
  });
});

describe('probePin timeoutMs', () => {
  it.each([0, -1, NaN, 1.5, Infinity, 3_600_001])(
    '%p is invalid_request, with no request sent',
    async (timeoutMs) => {
      const modelFor = jest.fn();
      await expect(
        probePin({ modelFor, pin: 'jev-1.13.0', timeoutMs })
      ).resolves.toEqual({ verified: false, reason: 'invalid_request' });
      expect(modelFor).not.toHaveBeenCalled();
    }
  );
});

describe('withFallback onAttempt', () => {
  const ranked = scripted('cross-encoder', () => ({
    results: [
      { index: 2, score: 3 },
      { index: 1, score: 2 },
      { index: 0, score: 1 },
    ],
    provider: 'cross-encoder',
    model: 'ce',
    status: 'ok',
    reason: null,
    attempts: [
      {
        provider: 'cross-encoder',
        model: 'ce',
        status: 'ok',
        reason: null,
        durationMs: 1,
        usage: { ...emptyUsage, calls: 1 },
      },
    ],
    durationMs: 1,
    usage: { ...emptyUsage, calls: 1 },
  }));

  it('sees a copy: a hook that mutates it changes nothing the chain returns', async () => {
    const later: Array<() => void> = [];
    const chain = withFallback([{ reranker: ranked }], {
      onAttempt: (_reranker, outcome) => {
        outcome.results.sort((a, b) => a.index - b.index);
        outcome.attempts[0].usage.calls = 99;
        later.push(() => {
          outcome.results.reverse();
          outcome.attempts.length = 0;
        });
      },
    });
    const outcome = await chain.rank({
      query: 'q',
      documents: ['a', 'b', 'c'],
    });
    for (const mutate of later) {
      mutate();
    }
    expect(outcome.status).toBe('ok');
    expect(outcome.results.map((entry) => entry.index)).toEqual([2, 1, 0]);
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.attempts[0].usage.calls).toBe(1);
    expect(outcome.usage.calls).toBe(1);
  });
});

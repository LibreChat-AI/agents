import http from 'node:http';
import type { AddressInfo } from 'node:net';
import type {
  DecisionModel,
  DecisionCredential,
  DecisionProviderSettings,
} from '@/decisions';
import type { RankRequest, RerankOutcome, OutcomeReranker } from './types';
import {
  probePin,
  rerankCost,
  noneReranker,
  withFallback,
  createScoreCache,
  toOutcomeReranker,
  createSystemOneReranker,
  createCrossEncoderReranker,
} from './core';
import {
  withBreaker,
  createBreaker,
  DecisionError,
  decisionPreset,
  createDecisionModel,
} from '@/decisions';

type Json = { [key: string]: unknown };

interface Call {
  path: string;
  body: Json;
  authorization?: string;
}

type Reply =
  | 'hang'
  | { status?: number; body?: unknown; raw?: string; delayMs?: number };

interface Stub {
  url: string;
  calls: Call[];
  maxConcurrent: number;
  close(): Promise<void>;
}

const servers: Stub[] = [];

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
        res.writeHead(answer.status ?? 200, {
          'content-type': 'application/json',
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
    expect(server.calls).toHaveLength(4);
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
    ]);
    expect(order(outcome)).toEqual([0, 1]);
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
});

const emptyUsage = {
  calls: 0,
  inputTokens: 0,
  outputTokens: 0,
  searchUnits: 0,
  cached: 0,
  estimatedTokens: 0,
  estimatedSearchUnits: 0,
};

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

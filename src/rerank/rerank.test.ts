import { createLogger } from 'winston';
import { ChatOpenAI } from '@langchain/openai';
import type { DecisionFetch, DecisionModel, DecisionResult } from '@/decisions';
import type { RerankObservation, SearchMetrics } from '@/tools/search/types';
import type { Reranker, RerankRequest, RerankResult } from './types';
import {
  createDecisionModel,
  decisionPreset,
  createStructuredChatDecisionModel,
} from '@/decisions';
import { createSystemOneReranker } from './systemone';
import { createWebSearchReranker } from './search';

const rubric = [
  'Not relevant',
  'Partly relevant',
  'Directly answers the query',
];

function fixture(answers: object) {
  const requests: object[] = [];
  const fetch: DecisionFetch = async (_url, init) => {
    requests.push(JSON.parse(init.body));
    return new Response(
      JSON.stringify({
        model: 'pinned-model',
        answers,
        usage: { input_tokens: 10, output_tokens: 2 },
      })
    );
  };
  const model = createDecisionModel(
    { ...decisionPreset('laya'), baseURL: 'http://localhost/v1/systemone' },
    undefined,
    { fetch, providerId: 'laya' }
  );
  return { model, requests };
}

const score = (value: number) => ({ type: 'score', score: value });

describe('System One reranking', () => {
  it('ranks one batch by expected rubric scores, preserving duplicate document identity and ties', async () => {
    const { model, requests } = fixture({
      d0: score(0.5),
      d1: score(2),
      d2: score(2),
    });
    const reranker = createSystemOneReranker({ model, rubric });
    const result = await reranker.rerank({
      query: 'query',
      documents: ['same', 'same', 'last'],
      topK: 2,
    });
    expect(result).toEqual({
      results: [
        { index: 1, score: 2 },
        { index: 2, score: 2 },
      ],
      model: 'pinned-model',
      usage: { inputTokens: 10, outputTokens: 2 },
    });
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      state: { query: 'query', documents: ['same', 'same', 'last'] },
      questions: {
        d0: { type: 'score', criteria: rubric },
        d1: { type: 'score', criteria: rubric },
        d2: { type: 'score', criteria: rubric },
      },
    });
  });

  it('rejects incomplete answers instead of assigning a missing candidate score zero', async () => {
    const { model } = fixture({ d1: score(1) });
    const reranker = createSystemOneReranker({ model, rubric });
    await expect(
      reranker.rerank({ query: 'query', documents: ['a', 'b'] })
    ).rejects.toMatchObject({ failure: 'malformed_response' });
  });

  it.each([{ documents: [] }, { documents: ['a'] }])(
    'avoids remote work for empty candidates or zero topK: %j',
    async ({ documents }) => {
      const { model, requests } = fixture({});
      const reranker = createSystemOneReranker({ model, rubric });
      expect(
        (await reranker.rerank({ query: 'q', documents, topK: 0 })).results
      ).toEqual([]);
      expect(requests).toHaveLength(0);
    }
  );

  it('rejects oversized batches locally without truncating the candidate set', async () => {
    const { model, requests } = fixture({});
    const reranker = createSystemOneReranker({
      model,
      rubric,
      maxDocuments: 1,
    });
    await expect(
      reranker.rerank({ query: 'q', documents: ['a', 'b'] })
    ).rejects.toMatchObject({ failure: 'unsupported_question' });
    expect(requests).toHaveLength(0);
  });

  it.each([-1, 0.5, NaN, Infinity])(
    'rejects invalid topK %s before transport',
    async (topK) => {
      const { model, requests } = fixture({});
      const reranker = createSystemOneReranker({ model, rubric });
      await expect(
        reranker.rerank({ query: 'q', documents: ['a'], topK })
      ).rejects.toMatchObject({ failure: 'bad_request' });
      expect(requests).toHaveLength(0);
    }
  );

  it('cannot fabricate expected scores through strict chat', async () => {
    const fetch = jest.fn(async (): Promise<Response> => {
      throw new Error('unexpected provider request');
    });
    const model = createStructuredChatDecisionModel({
      model: new ChatOpenAI({
        model: 'gpt-4o-mini',
        apiKey: 'test',
        configuration: { fetch },
      }),
      modelId: 'gpt-4o-mini',
      method: 'jsonSchema',
    });
    const reranker = createSystemOneReranker({ model, rubric });
    await expect(
      reranker.rerank({ query: 'q', documents: ['a'] })
    ).rejects.toMatchObject({ failure: 'unsupported_question' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('bounds non-cooperative decision models and propagates abort without leaking their errors', async () => {
    let signal: AbortSignal | undefined;
    const model: DecisionModel = {
      id: 'custom',
      model: 'custom',
      decide: async (request): Promise<DecisionResult> => {
        signal = request.signal;
        return new Promise(() => {});
      },
    };
    const reranker = createSystemOneReranker({ model, rubric, timeoutMs: 10 });
    await expect(
      reranker.rerank({ query: 'q', documents: ['a'] })
    ).rejects.toMatchObject({ failure: 'timeout' });
    expect(signal?.aborted).toBe(true);
    const controller = new AbortController();
    controller.abort();
    await expect(
      reranker.rerank({
        query: 'q',
        documents: ['a'],
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ failure: 'aborted' });
  });
});

function collector(observations: RerankObservation[]): SearchMetrics {
  return {
    recordRerank: (observation) => observations.push(observation),
    recordSearch: () => {},
    recordScrape: () => {},
    flush: () => {},
  };
}

const silentLogger = createLogger({ silent: true });

describe('web-search rerank adapter', () => {
  it.each([
    { timeoutMs: undefined, expectedScore: 0, aborted: true },
    { timeoutMs: 100, expectedScore: 2, aborted: false },
    { timeoutMs: 5, expectedScore: 0, aborted: true },
  ])(
    'preserves the ranker deadline unless explicitly overridden by $timeoutMs',
    async ({ timeoutMs, expectedScore, aborted }) => {
      jest.useFakeTimers();
      try {
        let signal: AbortSignal | undefined;
        const model: DecisionModel = {
          id: 'delayed',
          model: 'delayed',
          decide: async (request): Promise<DecisionResult> => {
            signal = request.signal;
            return new Promise((resolve) => {
              setTimeout(
                () =>
                  resolve({
                    answers: {
                      d0: {
                        type: 'score',
                        score: 2,
                        confidence: null,
                        probabilities: null,
                      },
                    },
                    model: 'delayed',
                    usage: null,
                  }),
                40
              );
            });
          },
        };
        const ranker = createSystemOneReranker({
          model,
          rubric,
          timeoutMs: 10,
        });
        const adapter = createWebSearchReranker(ranker, {
          timeoutMs,
          logger: silentLogger,
        });
        const observations: RerankObservation[] = [];
        const pending = adapter.rerank(
          'q',
          ['candidate'],
          1,
          collector(observations)
        );
        await jest.advanceTimersByTimeAsync(50);
        expect(await pending).toEqual([
          { text: 'candidate', score: expectedScore },
        ]);
        expect(signal?.aborted).toBe(aborted);
        expect(observations).toHaveLength(1);
        expect(observations[0].reason).toBe(aborted ? 'error' : undefined);
      } finally {
        jest.useRealTimers();
      }
    }
  );

  it('keeps the default outer deadline without forwarding a request override', async () => {
    jest.useFakeTimers();
    try {
      let signal: AbortSignal | undefined;
      const rerank = jest.fn(
        async (request: RerankRequest): Promise<RerankResult> => {
          signal = request.signal;
          return new Promise(() => {});
        }
      );
      const reranker: Reranker = { id: 'non-cooperative', rerank };
      const adapter = createWebSearchReranker(reranker, {
        logger: silentLogger,
      });
      const observations: RerankObservation[] = [];
      const pending = adapter.rerank(
        'q',
        ['candidate'],
        1,
        collector(observations)
      );
      await jest.advanceTimersByTimeAsync(10_001);
      expect(signal?.aborted).toBe(true);
      expect(rerank).toHaveBeenCalledTimes(1);
      expect(rerank.mock.calls[0][0]).not.toHaveProperty('timeoutMs');
      expect(await pending).toEqual([{ text: 'candidate', score: 0 }]);
      expect(observations).toHaveLength(1);
      expect(observations[0].reason).toBe('error');
    } finally {
      jest.useRealTimers();
    }
  });

  it('maps indices to text and records one metric while capping highlights', async () => {
    const reranker: Reranker = {
      id: 'host',
      rerank: async (): Promise<RerankResult> => ({
        results: [
          { index: 1, score: 0.8 },
          { index: 0, score: 0.4 },
        ],
        model: 'host-model',
        usage: { inputTokens: 12 },
      }),
    };
    const adapter = createWebSearchReranker(reranker, { logger: silentLogger });
    const observations: RerankObservation[] = [];
    expect(
      await adapter.rerank('q', ['first', 'second'], 1, collector(observations))
    ).toEqual([{ text: 'second', score: 0.8 }]);
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({
      provider: 'host',
      model: 'host-model',
      units: 12,
      results: 1,
    });
  });

  it.each([
    { results: [] },
    { results: [{ index: -1, score: 1 }] },
    { results: [{ index: 2, score: 1 }] },
    { results: [{ index: 0.5, score: 1 }] },
    { results: [{ index: 0, score: NaN }] },
    {
      results: [
        { index: 0, score: 1 },
        { index: 0, score: 0.5 },
      ],
    },
    {
      results: [
        { index: 0, score: 0.5 },
        { index: 1, score: 1 },
      ],
    },
  ])('falls back on invalid ranks %j', async ({ results }) => {
    const reranker: Reranker = {
      id: 'host',
      rerank: async () => ({ results, model: 'host', usage: null }),
    };
    const observations: RerankObservation[] = [];
    const adapter = createWebSearchReranker(reranker, { logger: silentLogger });
    expect(
      await adapter.rerank('q', ['first', 'second'], 2, collector(observations))
    ).toEqual([
      { text: 'first', score: 0 },
      { text: 'second', score: 0 },
    ]);
    expect(observations).toHaveLength(1);
    expect(observations[0].reason).toBeDefined();
  });

  it('returns original order on provider failures without exposing private text in metrics', async () => {
    const reranker: Reranker = {
      id: 'host',
      rerank: async () => {
        throw new Error('private key and document text');
      },
    };
    const observations: RerankObservation[] = [];
    const adapter = createWebSearchReranker(reranker, { logger: silentLogger });
    expect(
      await adapter.rerank('q', ['first'], 1, collector(observations))
    ).toEqual([{ text: 'first', score: 0 }]);
    expect(JSON.stringify(observations)).not.toContain('private');
  });
});

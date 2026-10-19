import vm from 'node:vm';
import {
  corpusStats,
  tokenizeCode,
  estimatedUsage,
  createBm25Reranker,
  ESTIMATED_TOKENS_PER_DOCUMENT,
} from './core';

describe('tokenizeCode', () => {
  it.each([
    ['parseHttpResponse', ['parse', 'http', 'response']],
    ['MAX_RETRY_COUNT', ['max', 'retry', 'count']],
    ['snake_case_name', ['snake', 'case', 'name']],
    ['kebab-case-name', ['kebab', 'case', 'name']],
    ['src/rerank/systemone.ts', ['src', 'rerank', 'systemone']],
    ['withFallback.deadlineMs', ['fallback', 'deadline']],
    ['`createBm25Reranker`', ['create', 'bm25', 'reranker']],
    ['How does the token refresh work?', ['token', 'refresh', 'work']],
    ['getID io x', ['get']],
  ])('splits %s into %j', (text, tokens) => {
    expect(tokenizeCode(text)).toEqual(tokens);
  });
});

const CORPUS = [
  'src/decisions/transport.ts\nretryAfterMs failureForStatus createTransport',
  'src/decisions/breaker.ts\nwithBreaker createBreaker isFatalDecisionError',
  'src/rerank/systemone.ts\ncreateSystemOneReranker rankDocuments',
  'src/rerank/fallback.ts\nwithFallback noneReranker runStep',
  'src/tools/search/rerankers.ts\nJinaReranker CohereReranker',
];

describe('corpusStats', () => {
  it('computes smoothed Okapi idf and the mean document length over the corpus', () => {
    const stats = corpusStats(CORPUS);
    expect(stats.n).toBe(5);
    const lengths = CORPUS.map((document) => tokenizeCode(document).length);
    expect(stats.avgdl).toBeCloseTo(
      lengths.reduce((sum, length) => sum + length, 0) / 5
    );
    expect(stats.idf.get('fallback')).toBeCloseTo(
      Math.log(1 + (5 - 1 + 0.5) / (1 + 0.5))
    );
    expect(stats.idf.get('rerank')).toBeCloseTo(
      Math.log(1 + (5 - 2 + 0.5) / (2 + 0.5))
    );
    expect(stats.idf.get('fallback')!).toBeGreaterThan(stats.idf.get('src')!);
    expect(corpusStats([])).toEqual({ idf: new Map(), avgdl: 0, n: 0 });
  });
});

describe('createBm25Reranker', () => {
  it('requires corpus statistics', () => {
    expect(() => createBm25Reranker({ idf: new Map(), avgdl: 0 })).toThrow(
      'corpusStats'
    );
    expect(() =>
      createBm25Reranker({
        idf: undefined as unknown as Map<string, number>,
        avgdl: 3,
      })
    ).toThrow('corpusStats');
  });

  it('accepts any ReadonlyMap of idf, including a Map from another realm', async () => {
    const stats = corpusStats(CORPUS);
    const wrapper: ReadonlyMap<string, number> = {
      get: (term) => stats.idf.get(term),
      has: (term) => stats.idf.has(term),
      forEach: (callback) => stats.idf.forEach(callback),
      entries: () => stats.idf.entries(),
      keys: () => stats.idf.keys(),
      values: () => stats.idf.values(),
      get size(): number {
        return stats.idf.size;
      },
      [Symbol.iterator]: () => stats.idf[Symbol.iterator](),
    };
    const foreign = vm.runInNewContext('new Map(entries)', {
      entries: [...stats.idf],
    }) as ReadonlyMap<string, number>;
    expect(foreign instanceof Map).toBe(false);
    const query = 'where is with_fallback defined';
    const expected = await createBm25Reranker(stats).rank({
      query,
      documents: CORPUS,
    });
    for (const idf of [wrapper, foreign]) {
      const outcome = await createBm25Reranker({
        idf,
        avgdl: stats.avgdl,
      }).rank({ query, documents: CORPUS });
      expect(outcome.status).toBe('ok');
      expect(outcome.results).toEqual(expected.results);
    }
  });

  it('ranks the lexical match first at no cost, splitting identifiers in the query and documents', async () => {
    const stats = corpusStats(CORPUS);
    const reranker = createBm25Reranker(stats);
    expect(reranker).toMatchObject({ id: 'bm25:okapi', kind: 'bm25' });
    const outcome = await reranker.rank({
      query: 'where is with_fallback defined',
      documents: CORPUS,
    });
    expect(outcome).toMatchObject({
      status: 'ok',
      provider: 'bm25',
      model: 'okapi',
      usage: { calls: 0, inputTokens: 0, estimatedTokens: 0 },
    });
    expect(outcome.results[0].index).toBe(3);
    expect(outcome.results[0].score).toBeGreaterThan(0);
    expect(outcome.results.slice(1).every((entry) => entry.score === 0)).toBe(
      true
    );
    expect(outcome.results.slice(1).map((entry) => entry.index)).toEqual([
      0, 1, 2, 4,
    ]);
  });

  it('gives a term the corpus never saw the idf of its rarest term', async () => {
    const reranker = createBm25Reranker({
      idf: new Map([
        ['common', 0.1],
        ['rare', 2],
      ]),
      avgdl: 2,
    });
    const outcome = await reranker.rank({
      query: 'unseen common',
      documents: ['common common', 'unseen'],
    });
    expect(outcome.results[0].index).toBe(1);
  });

  it('never throws: empty, invalid, and a failing tokenizer are outcomes', async () => {
    const reranker = createBm25Reranker({
      idf: () => 1,
      avgdl: 2,
      tokenize: (text) => {
        if (text === 'boom') {
          throw new Error('tokenizer failed');
        }
        return text.split(' ');
      },
    });
    expect((await reranker.rank({ query: 'q', documents: [] })).reason).toBe(
      'empty'
    );
    expect((await reranker.rank({ query: ' ', documents: ['a'] })).reason).toBe(
      'invalid_request'
    );
    expect(
      await reranker.rank({ query: 'q', documents: ['a', 'boom'] })
    ).toMatchObject({
      status: 'skipped',
      reason: 'error',
      results: [
        { index: 0, score: null },
        { index: 1, score: null },
      ],
    });
  });
});

describe('estimatedUsage', () => {
  it('charges abandoned System One requests per document and cross-encoder search units per 100 documents', () => {
    expect(estimatedUsage('systemone', 3)).toEqual({
      estimatedTokens: 3 * ESTIMATED_TOKENS_PER_DOCUMENT.systemone,
      estimatedSearchUnits: 0,
    });
    expect(estimatedUsage('cross-encoder', 101, 150)).toEqual({
      estimatedTokens: 101 * 150,
      estimatedSearchUnits: 2,
    });
    expect(estimatedUsage('bm25', 50)).toEqual({
      estimatedTokens: 0,
      estimatedSearchUnits: 0,
    });
    expect(estimatedUsage('none', 50)).toEqual({
      estimatedTokens: 0,
      estimatedSearchUnits: 0,
    });
    expect(estimatedUsage('systemone', -1)).toEqual({
      estimatedTokens: 0,
      estimatedSearchUnits: 0,
    });
  });
});

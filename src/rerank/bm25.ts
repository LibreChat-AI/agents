import type { RankRequest, RerankOutcome, OutcomeReranker } from './types';
import {
  stepOutcome,
  documentCount,
  emptyRerankUsage,
  rankRequestProblem,
} from './outcome';
import { DecisionError } from '../decisions/types';

const STOP_WORDS: ReadonlySet<string> = new Set(
  'which what where who how does that this the and for with from into over under of to in on at by an a is are be been do did as it its their they them or not no via when after before while than then also each all any some one two'.split(
    ' '
  )
);

/**
 * Codegraph's measured BM25 tokenizer for code and its prose: camelCase,
 * snake_case, kebab-case, dotted names, and paths split into words, which
 * are lower-cased; words of two letters or fewer and English stop words are
 * dropped.
 */
export function tokenizeCode(text: string): string[] {
  return text
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_\-./`]/g, ' ')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((word) => word.length > 2 && !STOP_WORDS.has(word));
}

export type Tokenize = (text: string) => string[];

export interface CorpusStats {
  /** Okapi idf, +1 smoothed so it is never negative. */
  idf: Map<string, number>;
  /** Mean document length in tokens. */
  avgdl: number;
  /** Documents counted. */
  n: number;
}

/** Statistics over the whole corpus a shortlist is drawn from, not over the shortlist. */
export function corpusStats(
  documents: Iterable<string>,
  options: { tokenize?: Tokenize } = {}
): CorpusStats {
  const tokenize = options.tokenize ?? tokenizeCode;
  const frequency = new Map<string, number>();
  let total = 0;
  let n = 0;
  for (const document of documents) {
    const tokens = tokenize(document);
    total += tokens.length;
    n += 1;
    for (const term of new Set(tokens)) {
      frequency.set(term, (frequency.get(term) ?? 0) + 1);
    }
  }
  const idf = new Map<string, number>();
  for (const [term, seen] of frequency) {
    idf.set(term, Math.log(1 + (n - seen + 0.5) / (seen + 0.5)));
  }
  return { idf, avgdl: n > 0 ? total / n : 0, n };
}

export interface Bm25RerankerOptions {
  /** Corpus idf from `corpusStats`, or a function of the term. */
  idf: ReadonlyMap<string, number> | ((term: string) => number);
  /** The corpus's mean document length in tokens. */
  avgdl: number;
  /** Defaults to 1.2. */
  k1?: number;
  /** Defaults to 0.75. */
  b?: number;
  /** Must match the tokenizer the statistics were built with. Defaults to `tokenizeCode`. */
  tokenize?: Tokenize;
}

/**
 * In-process Okapi BM25. Corpus statistics are required: idf over a 50-document
 * shortlist alone is close to degenerate. A term the corpus never saw counts
 * as rare as its rarest term. Never throws once built, and costs nothing.
 */
export function createBm25Reranker(
  options: Bm25RerankerOptions
): OutcomeReranker {
  const k1 = options.k1 ?? 1.2;
  const b = options.b ?? 0.75;
  const tokenize = options.tokenize ?? tokenizeCode;
  const { idf, avgdl } = options;
  let idfOf: ((term: string) => number) | undefined;
  if (idf instanceof Map) {
    let rarest = 0;
    for (const value of idf.values()) {
      rarest = Math.max(rarest, value);
    }
    idfOf = (term: string): number => idf.get(term) ?? rarest;
  } else if (typeof idf === 'function') {
    idfOf = idf;
  }
  if (
    idfOf === undefined ||
    typeof avgdl !== 'number' ||
    !Number.isFinite(avgdl) ||
    avgdl <= 0 ||
    !Number.isFinite(k1) ||
    k1 < 0 ||
    !Number.isFinite(b) ||
    b < 0 ||
    b > 1 ||
    typeof tokenize !== 'function'
  ) {
    throw new DecisionError(
      'bad_request',
      'bm25 reranker requires corpus statistics (corpusStats)',
      { provider: 'bm25' }
    );
  }
  const termIdf = idfOf;
  const kind = 'bm25' as const;
  const model = 'okapi';

  return {
    id: 'bm25:okapi',
    kind,
    model,
    async rank(request: RankRequest): Promise<RerankOutcome> {
      const startedAt = performance.now();
      const documents = documentCount(request);
      const base = { kind, documents, startedAt, model };
      const problem = rankRequestProblem(request);
      if (problem !== undefined) {
        return stepOutcome({ ...base, reason: problem, called: false });
      }
      try {
        const terms = tokenize(request.query);
        const scores = request.documents.map((document) => {
          const tokens = tokenize(document);
          const frequency = new Map<string, number>();
          for (const token of tokens) {
            frequency.set(token, (frequency.get(token) ?? 0) + 1);
          }
          const norm = k1 * (1 - b + (b * tokens.length) / avgdl);
          let score = 0;
          for (const term of terms) {
            const seen = frequency.get(term);
            if (seen !== undefined) {
              score += (termIdf(term) * (seen * (k1 + 1))) / (seen + norm);
            }
          }
          return Number.isFinite(score) ? score : null;
        });
        if (scores.includes(null)) {
          return stepOutcome({ ...base, reason: 'error' });
        }
        return stepOutcome({ ...base, scores, usage: emptyRerankUsage() });
      } catch {
        return stepOutcome({ ...base, reason: 'error' });
      }
    },
  };
}

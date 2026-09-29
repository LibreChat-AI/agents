import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger, formatErrorForLog } from './utils';
import { createSearchMetrics } from './metrics';

const DEFAULT_JINA_API_URL = 'https://api.jina.ai/v1/rerank';
const DEFAULT_COHERE_API_URL = 'https://api.cohere.com/v2/rerank';

/** Every other network call in the search pipeline is bounded (scrapers,
 * search providers); rerank requests must be too, or a hung rerank API
 * stalls the whole tool. */
const DEFAULT_RERANKER_TIMEOUT = 10000;

const getDefaultJinaApiUrl = (): string =>
  process.env.JINA_API_URL != null && process.env.JINA_API_URL !== ''
    ? process.env.JINA_API_URL
    : DEFAULT_JINA_API_URL;

const getDefaultCohereApiUrl = (): string =>
  process.env.COHERE_API_URL != null && process.env.COHERE_API_URL !== ''
    ? process.env.COHERE_API_URL
    : DEFAULT_COHERE_API_URL;

/**
 * Ranking used whenever no reranker scores the chunks: the candidates' own
 * order, capped at `topK`, with a neutral score.
 *
 * Exported because it is not only a *fallback*. With `rerankerType: 'none'`
 * there is no reranker to fall back from, and the search pipeline needs the
 * same ranking to hand the scraped text downstream — see `getHighlights` in
 * `./search`.
 */
export const getDefaultRanking = (
  documents: string[],
  topK: number
): t.Highlight[] =>
  documents
    .slice(0, Math.min(topK, documents.length))
    .map((doc) => ({ text: doc, score: 0 }));

export abstract class BaseReranker {
  protected apiKey: string | undefined;
  protected logger: t.Logger;
  /** Public so a caller that fails before reaching `rerank` can still
   * attribute the attempt to the configured reranker. */
  abstract readonly provider: string;
  private ownMetrics?: t.SearchMetrics;

  constructor(logger?: t.Logger) {
    // Each specific reranker will set its API key
    this.logger = logger || createDefaultLogger();
  }

  /**
   * A search reranks once per scraped source, so nothing here logs per call:
   * every exit records one observation through `metrics` and the enclosing
   * search emits a single summary. `metrics` is optional only for direct use
   * of a reranker, which falls back to an auto-flushing collector.
   */
  abstract rerank(
    query: string,
    documents: string[],
    topK?: number,
    metrics?: t.SearchMetrics
  ): Promise<t.Highlight[]>;

  protected getDefaultRanking(
    documents: string[],
    topK: number
  ): t.Highlight[] {
    return getDefaultRanking(documents, topK);
  }

  /** A direct caller has no enclosing search to fold into, so one
   * auto-flushing collector per instance emits that call's summary on its
   * own; record and flush run synchronously, so concurrent calls on the same
   * instance cannot share a total. */
  private localMetrics(): t.SearchMetrics {
    this.ownMetrics ??= createSearchMetrics(this.logger, true);
    return this.ownMetrics;
  }

  /** Opens the per-call state every exit path records against. */
  protected beginRerank(
    documents: string[],
    topK: number,
    metrics?: t.SearchMetrics
  ): t.RerankRun {
    return {
      metrics: metrics ?? this.localMetrics(),
      documents,
      topK,
      startedAt: Date.now(),
    };
  }

  private record(
    run: t.RerankRun,
    highlights: t.Highlight[],
    reason?: t.RerankFallback,
    error?: t.SafeErrorLog
  ): t.Highlight[] {
    run.metrics.recordRerank({
      provider: this.provider,
      chunks: run.documents.length,
      results: highlights.length,
      durationMs: Date.now() - run.startedAt,
      model: run.model,
      units: run.units,
      dropped: run.dropped,
      topK: run.topKLimit != null ? run.topK : undefined,
      topKLimit: run.topKLimit,
      reason,
      error,
    });
    return highlights;
  }

  protected complete(
    run: t.RerankRun,
    highlights: t.Highlight[]
  ): t.Highlight[] {
    return this.record(run, highlights);
  }

  /** Records the call as a fallback and returns the candidates' input order. */
  protected fallback(
    run: t.RerankRun,
    reason: t.RerankFallback,
    error?: t.SafeErrorLog
  ): t.Highlight[] {
    return this.record(
      run,
      this.getDefaultRanking(run.documents, run.topK),
      reason,
      error
    );
  }
}

export class JinaReranker extends BaseReranker {
  readonly provider = 'jina';
  private apiUrl: string;
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    apiKey = process.env.JINA_API_KEY,
    apiUrl = getDefaultJinaApiUrl(),
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    apiKey?: string;
    apiUrl?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5,
    metrics?: t.SearchMetrics
  ): Promise<t.Highlight[]> {
    const run = this.beginRerank(documents, topK, metrics);

    try {
      if (this.apiKey == null || this.apiKey === '') {
        return this.fallback(run, 'no_api_key');
      }

      const requestData = {
        model: 'jina-reranker-v2-base-multilingual',
        query: query,
        top_n: topK,
        documents: documents,
        return_documents: true,
      };

      const response = await axios.post<t.JinaRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: this.timeout,
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
        }
      );

      run.model = response.data?.model;
      run.units = response.data?.usage?.total_tokens;

      const results = response.data?.results;
      if (!Array.isArray(results) || results.length === 0) {
        return this.fallback(run, 'bad_response');
      }

      return this.complete(
        run,
        results.map((result) => {
          const docIndex = result.index;
          const score = result.relevance_score;
          let text = '';

          // If return_documents is true, the document field will be present
          if (result.document != null) {
            const doc = result.document;
            if (typeof doc === 'object' && 'text' in doc) {
              text = doc.text;
            } else if (typeof doc === 'string') {
              text = doc;
            }
          } else {
            // Otherwise, use the index to get the document
            text = documents[docIndex];
          }

          return { text, score };
        })
      );
    } catch (error) {
      return this.fallback(run, 'error', formatErrorForLog(error));
    }
  }
}

export class CohereReranker extends BaseReranker {
  readonly provider = 'cohere';
  private apiUrl: string;
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    apiKey = process.env.COHERE_API_KEY,
    apiUrl = getDefaultCohereApiUrl(),
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    apiKey?: string;
    apiUrl?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5,
    metrics?: t.SearchMetrics
  ): Promise<t.Highlight[]> {
    const run = this.beginRerank(documents, topK, metrics);

    try {
      if (this.apiKey == null || this.apiKey === '') {
        return this.fallback(run, 'no_api_key');
      }

      const model = 'rerank-v3.5';
      const requestData = {
        model,
        query: query,
        top_n: topK,
        documents: documents,
      };

      const response = await axios.post<t.CohereRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: this.timeout,
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
        }
      );

      run.model = model;
      run.units = response.data?.meta?.billed_units?.search_units;

      const results = response.data?.results;
      if (!Array.isArray(results) || results.length === 0) {
        return this.fallback(run, 'bad_response');
      }

      return this.complete(
        run,
        results.map((result) => {
          const docIndex = result.index;
          const score = result.relevance_score;
          const text = documents[docIndex];
          return { text, score };
        })
      );
    } catch (error) {
      return this.fallback(run, 'error', formatErrorForLog(error));
    }
  }
}

/** rag_api's `/v1/rerank` contract caps candidates at 50 and `top_n` at 25;
 * violating either is a client bug, not a server error, so both are clamped
 * locally with a debug log instead of erroring. */
const RAG_API_MAX_CANDIDATES = 50;
const RAG_API_MAX_TOP_N = 25;
const RAG_API_DEFAULT_PROFILE = 'fast-v1';

const getDefaultRagApiUrl = (): string | undefined =>
  process.env.RAG_API_URL != null && process.env.RAG_API_URL !== ''
    ? process.env.RAG_API_URL
    : undefined;

const toRagApiCandidate = (
  text: string,
  index: number
): t.RagApiRerankCandidate => ({ id: String(index), text, base_score: 0 });

/** A single source split into overlapping chunks routinely exceeds the
 * contract limit, so documents are capped before any candidate is built:
 * only the submittable window is ever allocated. The overflow is reported
 * through the run's `dropped` counter rather than a per-call log. */
const buildRagApiCandidates = (
  documents: string[]
): t.RagApiRerankCandidate[] =>
  documents.length <= RAG_API_MAX_CANDIDATES
    ? documents.map(toRagApiCandidate)
    : documents.slice(0, RAG_API_MAX_CANDIDATES).map(toRagApiCandidate);

/** Reports the clamp on `run` rather than logging it per call: a search
 * configured above the contract limit returns fewer highlights than asked
 * for, and the summary is the only place that now says so. */
const clampRagApiTopN = (topN: number, run: t.RerankRun): number => {
  if (topN <= RAG_API_MAX_TOP_N) {
    return topN;
  }
  run.topKLimit = RAG_API_MAX_TOP_N;
  return RAG_API_MAX_TOP_N;
};

/** Deterministic tie ordering: rank by score descending, breaking ties on
 * original candidate index rather than relying on rag_api's response order. */
const sortRagApiResults = (
  results: t.RagApiRerankResult[]
): t.RagApiRerankResult[] =>
  [...results].sort((a, b) => b.score - a.score || a.index - b.index);

/** Indices are only meaningful against the candidates actually submitted:
 * anything beyond `candidateCount` refers to text the server never saw. */
const isValidRagApiResult = (
  result: t.RagApiRerankResult | undefined,
  candidateCount: number
): result is t.RagApiRerankResult =>
  result != null &&
  typeof result.index === 'number' &&
  Number.isInteger(result.index) &&
  result.index >= 0 &&
  result.index < candidateCount &&
  typeof result.score === 'number' &&
  Number.isFinite(result.score);

/** A repeated index would map one document into several `top_n` slots and
 * silently drop distinct results, so a duplicate invalidates the batch just
 * like any other malformed row. Seen indices are tracked in the same pass
 * that validates each row. */
const isValidRagApiBatch = (
  results: t.RagApiRerankResult[],
  candidateCount: number
): boolean => {
  const seenIndices = new Set<number>();
  return results.every((result) => {
    if (!isValidRagApiResult(result, candidateCount)) {
      return false;
    }
    if (seenIndices.has(result.index)) {
      return false;
    }
    seenIndices.add(result.index);
    return true;
  });
};

/** Bounds the whole rerank round trip, token acquisition included: the
 * supplier mints its token over the network, so awaiting it before axios
 * starts would leave the search unbounded whenever an auth service stalls.
 * The signal reaches both legs, so returning a fallback also cancels whatever
 * is still in flight rather than leaving a request running past its caller.
 * A non-positive timeout keeps axios' "no timeout" semantics. */
const withRerankDeadline = <T>(
  operation: (signal?: AbortSignal) => Promise<T>,
  timeout: number
): Promise<T> => {
  if (timeout <= 0) {
    return operation();
  }

  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error(`rag_api rerank exceeded its ${timeout}ms timeout.`));
    }, timeout);
  });

  const pending = operation(controller.signal);
  pending.catch(() => undefined);

  return Promise.race([pending, deadline]).finally(() => clearTimeout(timer));
};

/**
 * Calls the public `danny-avila/rag_api` `/v1/rerank` endpoint (the
 * `fast-v1` profile). Auth is supplied per call by a token supplier (a
 * short-lived JWT minted by the host app) rather than a static key. A
 * reranker failure must never throw into the search flow: on a missing
 * base URL/token supplier, a timeout (covering token acquisition as well as
 * the request), a non-2xx response, or any malformed payload, this falls back
 * to the candidates' original order via {@link BaseReranker.getDefaultRanking}.
 */
export class RagApiReranker extends BaseReranker {
  readonly provider = 'rag-api';
  private baseUrl?: string;
  private tokenSupplier?: t.RagApiTokenSupplier;
  private profile: string;
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    baseUrl = getDefaultRagApiUrl(),
    tokenSupplier,
    profile = RAG_API_DEFAULT_PROFILE,
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    baseUrl?: string;
    tokenSupplier?: t.RagApiTokenSupplier;
    profile?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.baseUrl = baseUrl?.replace(/\/+$/, '');
    this.tokenSupplier = tokenSupplier;
    this.profile = profile;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5,
    metrics?: t.SearchMetrics
  ): Promise<t.Highlight[]> {
    if (documents.length === 0) {
      return [];
    }

    const run = this.beginRerank(documents, topK, metrics);

    const baseUrl = this.baseUrl;
    if (baseUrl == null || baseUrl === '') {
      return this.fallback(run, 'no_base_url');
    }

    const tokenSupplier = this.tokenSupplier;
    if (tokenSupplier == null) {
      return this.fallback(run, 'no_token_supplier');
    }

    const candidates = buildRagApiCandidates(documents);
    const topN = clampRagApiTopN(Math.max(0, topK), run);
    run.dropped = documents.length - candidates.length;
    const requestData: t.RagApiRerankRequestBody = {
      profile: this.profile,
      query,
      candidates,
      top_n: topN,
    };

    try {
      const data = await withRerankDeadline(async (signal) => {
        const token = await tokenSupplier(signal);
        const response = await axios.post<t.RagApiRerankResponse | undefined>(
          `${baseUrl}/v1/rerank`,
          requestData,
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
            timeout: this.timeout,
            httpAgent: this.httpAgent,
            httpsAgent: this.httpsAgent,
            signal,
          }
        );
        return response.data;
      }, this.timeout);

      run.model = data?.model;

      const rawResults = data?.results;
      if (!Array.isArray(rawResults) || rawResults.length === 0) {
        return this.fallback(run, 'bad_response');
      }

      if (!isValidRagApiBatch(rawResults, candidates.length)) {
        return this.fallback(run, 'invalid_results');
      }

      return this.complete(
        run,
        sortRagApiResults(rawResults)
          .slice(0, topN)
          .map((result) => ({
            text: documents[result.index],
            score: result.score,
          }))
      );
    } catch (error) {
      return this.fallback(run, 'error', formatErrorForLog(error));
    }
  }
}

export class InfinityReranker extends BaseReranker {
  readonly provider = 'infinity';

  constructor(logger?: t.Logger) {
    super(logger);
    // No API key needed for the placeholder implementation
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5,
    metrics?: t.SearchMetrics
  ): Promise<t.Highlight[]> {
    // This would be replaced with actual Infinity reranker implementation
    return this.fallback(
      this.beginRerank(documents, topK, metrics),
      'placeholder'
    );
  }
}

export class CustomReranker extends BaseReranker {
  private apiUrl: string | undefined;
  private model: string | undefined;
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    apiUrl = process.env.CUSTOM_RERANKER_API_URL,
    apiKey = process.env.CUSTOM_RERANKER_API_KEY,
    model = process.env.CUSTOM_RERANKER_MODEL,
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    apiUrl?: string;
    apiKey?: string;
    model?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.model = model;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(
      `Reranking ${documents.length} chunks with custom reranker using API URL: ${this.apiUrl}`
    );

    if (this.apiUrl == null || this.apiUrl === '') {
      this.logger.warn(
        'CUSTOM_RERANKER_API_URL is not set. Using default ranking.'
      );
      return this.getDefaultRanking(documents, topK);
    }

    if (this.model == null || this.model === '') {
      this.logger.warn(
        'CUSTOM_RERANKER_MODEL is not set. Using default ranking.'
      );
      return this.getDefaultRanking(documents, topK);
    }

    try {
      const requestData = {
        model: this.model,
        query: query,
        top_n: topK,
        documents: documents,
        return_documents: true,
      };

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey != null && this.apiKey !== '') {
        headers.Authorization = `Bearer ${this.apiKey}`;
      }

      const response = await axios.post<t.JinaRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        {
          headers,
          timeout: this.timeout,
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
        }
      );

      this.logger.debug('Custom Reranker API Model:', response.data?.model);
      this.logger.debug('Custom Reranker API Usage:', response.data?.usage);

      if (response.data && response.data.results.length) {
        return response.data.results.map((result) => {
          const docIndex = result.index;
          const score = result.relevance_score;
          let text = '';

          if (result.document != null) {
            const doc = result.document;
            if (typeof doc === 'object' && 'text' in doc) {
              text = doc.text;
            } else if (typeof doc === 'string') {
              text = doc;
            }
          } else {
            text = documents[docIndex];
          }

          return { text, score };
        });
      } else {
        this.logger.warn(
          'Unexpected response format from custom reranker API. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }
    } catch (error) {
      this.logger.error(
        'Error using custom reranker',
        formatErrorForLog(error)
      );
      return this.getDefaultRanking(documents, topK);
    }
  }
}

/**
 * Creates the appropriate reranker based on type and configuration
 */
export const createReranker = (
  config: {
    rerankerType: t.RerankerType;
    jinaApiKey?: string;
    jinaApiUrl?: string;
    cohereApiKey?: string;
    cohereApiUrl?: string;
    ragApiUrl?: string;
    ragApiTokenSupplier?: t.RagApiTokenSupplier;
    ragApiProfile?: string;
    customRerankerApiUrl?: string;
    customRerankerApiKey?: string;
    customRerankerModel?: string;
    rerankerTimeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig
): BaseReranker | undefined => {
  const {
    rerankerType,
    jinaApiKey,
    jinaApiUrl,
    cohereApiKey,
    cohereApiUrl,
    ragApiUrl,
    ragApiTokenSupplier,
    ragApiProfile,
    customRerankerApiUrl,
    customRerankerApiKey,
    customRerankerModel,
    rerankerTimeout,
    logger,
    httpAgent,
    httpsAgent,
  } = config;

  // Create a default logger if none is provided
  const defaultLogger = logger || createDefaultLogger();

  switch (rerankerType.toLowerCase()) {
  case 'jina':
    return new JinaReranker({
      apiKey: jinaApiKey,
      apiUrl: jinaApiUrl,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  case 'cohere':
    return new CohereReranker({
      apiKey: cohereApiKey,
      apiUrl: cohereApiUrl,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  case 'rag-api':
    return new RagApiReranker({
      baseUrl: ragApiUrl,
      tokenSupplier: ragApiTokenSupplier,
      profile: ragApiProfile,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  case 'custom':
    return new CustomReranker({
      apiUrl: customRerankerApiUrl,
      apiKey: customRerankerApiKey,
      model: customRerankerModel,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  case 'infinity':
    return new InfinityReranker(defaultLogger);
  case 'none':
    defaultLogger.debug('Skipping reranking as reranker is set to "none"');
    return undefined;
  default:
    defaultLogger.warn(
      `Unknown reranker type: ${rerankerType}. Defaulting to InfinityReranker.`
    );
    return new JinaReranker({
      apiKey: jinaApiKey,
      apiUrl: jinaApiUrl,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  }
};

// Example usage:
// const jinaReranker = new JinaReranker();
// const cohereReranker = new CohereReranker();
// const infinityReranker = new InfinityReranker();

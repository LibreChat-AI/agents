import type { RankRequest, RerankOutcome, OutcomeReranker } from './types';
import type { DecisionCredential } from '../decisions/types';
import type { DecisionFetch } from '../decisions/transport';
import {
  isAborted,
  stepOutcome,
  documentCount,
  validDeadline,
  estimatedUsage,
  rerankReasonOf,
  emptyRerankUsage,
  rankRequestProblem,
} from './outcome';
import { DecisionError, isDecisionObject } from '../decisions/types';
import { createTransport } from '../decisions/transport';

export interface CrossEncoderRerankerOptions {
  /**
   * Full URL of a Cohere-shaped rerank endpoint: Cohere's `/v2/rerank`,
   * Jina's `/v1/rerank`, an Infinity server, or the ClickHouse gateway's
   * `${GATEWAY_URL}/rerank` (no `/v1`), which serves `qwen3-reranker-8b` and
   * `cohere-rerank-v3-5`. TEI's `/rerank` takes `texts` and answers a bare
   * array, so it does not fit. A response is capped at 256 KiB, so a host
   * that echoes each document back (Jina does by default) fails a large
   * shortlist as `bad_response`. Without a URL, every ranking is skipped
   * with `no_endpoint`.
   */
  url?: string;
  model: string;
  /** A bearer key or token minter; a 401 mints one fresh token for that call. */
  apiKey?: DecisionCredential;
  /** Defaults to true; without a key every ranking is skipped with `no_api_key`. */
  requiresAuth?: boolean;
  /** Defaults to 30 seconds; a request's `deadlineMs` overrides it. */
  timeoutMs?: number;
  /** Retries for rate limits, server errors, and network failures. Defaults to 0. */
  maxRetries?: number;
  /** Defaults to 1,000, Cohere's limit; larger sets are skipped, never truncated. */
  maxDocuments?: number;
  /** The estimate charged for a request a deadline abandoned. Defaults to 200. */
  estimatedTokensPerDocument?: number;
  fetch?: DecisionFetch;
}

type Parsed =
  | {
      ok: true;
      scores: number[];
      model: string | null;
      tokens: number;
      units: number;
    }
  | {
      ok: false;
      reason: 'bad_response' | 'invalid_results';
      model: string | null;
      tokens: number;
      units: number;
    };

const nonNegative = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;

/** Reads `results[{ index, relevance_score }]`, which must score every document exactly once. */
function parseResponse(body: string, documents: number): Parsed {
  let json: unknown;
  try {
    json = JSON.parse(body);
  } catch {
    return {
      ok: false,
      reason: 'bad_response',
      model: null,
      tokens: 0,
      units: 0,
    };
  }
  if (!isDecisionObject(json)) {
    return {
      ok: false,
      reason: 'bad_response',
      model: null,
      tokens: 0,
      units: 0,
    };
  }
  const meta = isDecisionObject(json.meta) ? json.meta : {};
  const tokens = isDecisionObject(meta.tokens) ? meta.tokens : {};
  const billed = isDecisionObject(meta.billed_units) ? meta.billed_units : {};
  const usage = isDecisionObject(json.usage) ? json.usage : {};
  const read = {
    model:
      typeof json.model === 'string' && json.model !== '' ? json.model : null,
    tokens:
      nonNegative(tokens.input_tokens) ||
      nonNegative(usage.prompt_tokens) ||
      nonNegative(usage.total_tokens),
    units: nonNegative(billed.search_units),
  };
  const list = json.results;
  if (!Array.isArray(list) || list.length !== documents) {
    return { ok: false, reason: 'invalid_results', ...read };
  }
  const scores: Array<number | undefined> = new Array(documents).fill(
    undefined
  );
  for (const entry of list as unknown[]) {
    if (!isDecisionObject(entry)) {
      return { ok: false, reason: 'invalid_results', ...read };
    }
    const { index, relevance_score: score } = entry;
    if (
      typeof index !== 'number' ||
      !Number.isSafeInteger(index) ||
      index < 0 ||
      index >= documents ||
      scores[index] !== undefined ||
      typeof score !== 'number' ||
      !Number.isFinite(score)
    ) {
      return { ok: false, reason: 'invalid_results', ...read };
    }
    scores[index] = score;
  }
  return { ok: true, scores: scores as number[], ...read };
}

/**
 * A Cohere-shaped rerank endpoint: one `POST { model, query, documents }`
 * answered by `results[{ index, relevance_score }]`. Never throws. The
 * response must score every document exactly once; scores are the host's and
 * carry no calibrated threshold, so `kept` is never set. Requests do not
 * follow redirects, and a deadline that fires after the request was sent is
 * charged at `estimatedUsage`.
 */
export function createCrossEncoderReranker(
  options: CrossEncoderRerankerOptions
): OutcomeReranker {
  const model = typeof options.model === 'string' ? options.model.trim() : '';
  const url = typeof options.url === 'string' ? options.url.trim() : '';
  const timeoutMs = options.timeoutMs ?? 30_000;
  const maxRetries = options.maxRetries ?? 0;
  const maxDocuments = options.maxDocuments ?? 1_000;
  const { apiKey } = options;
  const requiresAuth = options.requiresAuth !== false;
  if (
    model === '' ||
    validDeadline(timeoutMs) === undefined ||
    !Number.isSafeInteger(maxRetries) ||
    maxRetries < 0 ||
    maxRetries > 5 ||
    !Number.isSafeInteger(maxDocuments) ||
    maxDocuments < 1
  ) {
    throw new DecisionError('bad_request', 'invalid reranker configuration', {
      provider: 'cross-encoder',
    });
  }
  const id = `cross-encoder:${model}`;
  const kind = 'cross-encoder' as const;

  return {
    id,
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
      if (documents > maxDocuments) {
        return stepOutcome({
          ...base,
          reason: 'too_many_documents',
          called: false,
        });
      }
      if (url === '') {
        return stepOutcome({ ...base, reason: 'no_endpoint', called: false });
      }
      if (
        requiresAuth &&
        (apiKey == null || (typeof apiKey === 'string' && !apiKey.trim()))
      ) {
        return stepOutcome({ ...base, reason: 'no_api_key', called: false });
      }
      if (isAborted(request.signal)) {
        return stepOutcome({ ...base, reason: 'aborted', called: false });
      }
      const usage = emptyRerankUsage();
      const fetchImpl: DecisionFetch = options.fetch ?? globalThis.fetch;
      const counting: DecisionFetch = (input, init) => {
        usage.calls += 1;
        return fetchImpl(input, init);
      };
      try {
        const send = createTransport({
          providerId: id,
          apiKey,
          requiresAuth,
          endpoint: url,
          timeoutMs: validDeadline(request.deadlineMs) ?? timeoutMs,
          maxRetries,
          fetch: counting,
        });
        const parsed = await send(
          () => ({
            payload: JSON.stringify({
              model,
              query: request.query,
              documents: [...request.documents],
            }),
            parse: (body: string): Parsed => parseResponse(body, documents),
          }),
          request.signal,
          'rerank'
        );
        usage.inputTokens = parsed.tokens;
        usage.searchUnits = parsed.units;
        const said = parsed.model ?? model;
        if (!parsed.ok) {
          return stepOutcome({
            ...base,
            model: said,
            reason: parsed.reason,
            usage,
          });
        }
        return stepOutcome({
          ...base,
          model: said,
          scores: parsed.scores,
          usage,
        });
      } catch (error) {
        const reason = isAborted(request.signal)
          ? 'aborted'
          : rerankReasonOf(error);
        if (usage.calls > 0 && (reason === 'timeout' || reason === 'aborted')) {
          Object.assign(
            usage,
            estimatedUsage(kind, documents, options.estimatedTokensPerDocument)
          );
        }
        return stepOutcome({ ...base, reason, usage });
      }
    },
  };
}

import type { DecisionUsage } from '../decisions/types';

export interface RerankRequest {
  query: string;
  documents: readonly string[];
  topK?: number;
  signal?: AbortSignal;
  timeoutMs?: number;
  /**
   * Called once per request a reranker sends, retries included, for one that
   * reports them (see `DecisionRequest.onRequest`). Not awaited; a throw is ignored.
   */
  onRequest?: () => void;
  /**
   * Called once per reported request that ended with a reply (its HTTP
   * status, and its `Retry-After` in milliseconds when it was not OK and
   * sent one) or a failure without one (`null`), for one that reports them
   * (see `DecisionRequest.onResponse`). Not awaited; a throw is ignored.
   */
  onResponse?: (status: number | null, retryAfterMs?: number) => void;
}

export interface RankedDocument {
  /** Index into the submitted documents, not an identity derived from text. */
  index: number;
  /** Provider-specific relevance score, ordered descending; not a portable threshold. */
  score: number;
}

export interface RerankResult {
  results: RankedDocument[];
  model: string;
  usage: DecisionUsage | null;
}

/** Rankings retain candidate identity; failures reject rather than fabricate scores. */
export interface Reranker {
  readonly id: string;
  /**
   * True when `rerank` reports every request it sends through `onRequest` and
   * every one that ends through `onResponse` (see
   * `DecisionModel.reportsRequests`), so `toOutcomeReranker` counts only what
   * was really sent.
   */
  readonly reportsRequests?: boolean;
  rerank(request: RerankRequest): Promise<RerankResult>;
}

/**
 * Where a ranking came from. `systemone` asks a Jev/Laya decision model,
 * `cross-encoder` a Cohere-shaped `/rerank` endpoint, `bm25` scores in
 * process, and `none` is identity order.
 */
export type RerankKind = 'systemone' | 'cross-encoder' | 'bm25' | 'none';

/** `ok`: the first step answered. `degraded`: a later step did. `skipped`: identity order. */
export type RerankStatus = 'ok' | 'degraded' | 'skipped';

/**
 * Reasons the SDK reports. A `withFallback` gate may report its own string
 * (for example `backoff` or `budget`), so outcome reasons are typed `string`.
 */
export type RerankReason =
  | 'empty'
  | 'invalid_request'
  | 'too_many_documents'
  | 'disabled'
  | 'no_endpoint'
  | 'no_api_key'
  | 'timeout'
  | 'aborted'
  | 'partial'
  | 'unauthorized'
  | 'quota'
  | 'rate_limited'
  | 'circuit_open'
  | 'unsupported'
  | 'pin_mismatch'
  | 'bad_response'
  | 'invalid_results'
  | 'error';

/**
 * What a ranking was charged. Reported counts are summed as the host sent
 * them; `estimated*` counts charge requests a host received but a deadline
 * abandoned (see `estimatedUsage`): for a reranker whose requests are
 * reported, only those still in flight when it fired, never one the host had
 * already answered. Never a price.
 */
export interface RerankUsage {
  /**
   * Requests sent, including re-issues and retries. A decision model that
   * reports its requests (`reportsRequests`, as the SDK's HTTP decision model
   * does) is counted from those reports, so a `decide()` that failed before
   * sending (a credential minter that failed or hung, a state that cannot be
   * serialized) counts nothing; any other model counts one per `decide()`.
   */
  calls: number;
  inputTokens: number;
  outputTokens: number;
  searchUnits: number;
  /** Scores served from a score cache instead of a request. */
  cached: number;
  estimatedTokens: number;
  estimatedSearchUnits: number;
}

/** One step that ran (`ok` or `failed`) or was skipped without a request. */
export interface RerankAttempt {
  provider: RerankKind;
  model: string;
  status: 'ok' | 'failed' | 'skipped';
  reason: string | null;
  durationMs: number;
  usage: RerankUsage;
}

export interface RerankEntry {
  /** Index into the submitted documents. */
  index: number;
  /** Provider-specific score; `null` when the document was not scored. */
  score: number | null;
  /** Set only by a step with a threshold calibrated for its pinned model. */
  kept?: boolean;
}

/**
 * A ranking that never rejects. `results` is always a full permutation of the
 * input: score descending, ties in input order, `null` scores last. A
 * `skipped` outcome is identity order with every score `null`, so a caller
 * can always serve the order it already had.
 */
export interface RerankOutcome {
  results: RerankEntry[];
  provider: RerankKind;
  /** The model the host echoed, else the configured one. */
  model: string;
  status: RerankStatus;
  /** `null` when `ok`; for `degraded` and `skipped`, the first failed step's reason. */
  reason: string | null;
  attempts: RerankAttempt[];
  durationMs: number;
  /** Summed over every attempt: spend counts what was sent, not what answered. */
  usage: RerankUsage;
  /** Present when the answering step has a pin: whether every response echoed it. */
  pinVerified?: boolean;
  /** The calibrated threshold behind `kept`, when the answering step has one. */
  threshold?: number;
}

/** No `topK`: an outcome ranks every document, and the caller slices. */
export interface RankRequest {
  query: string;
  documents: readonly string[];
  signal?: AbortSignal;
  /**
   * This call's deadline. It caps the reranker's own timeout rather than
   * replacing it: a reranker configured with a shorter timeout still gives up
   * at that timeout, so a chain's later steps keep their time.
   */
  deadlineMs?: number;
}

/** Ranks without throwing or rejecting; every failure is an outcome's `reason`. */
export interface OutcomeReranker {
  readonly id: string;
  readonly kind: RerankKind;
  readonly model: string;
  rank(request: RankRequest): Promise<RerankOutcome>;
}

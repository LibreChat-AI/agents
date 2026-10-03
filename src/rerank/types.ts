import type { DecisionUsage } from '@/decisions';

export interface RerankRequest {
  query: string;
  documents: readonly string[];
  topK?: number;
  signal?: AbortSignal;
  timeoutMs?: number;
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
  rerank(request: RerankRequest): Promise<RerankResult>;
}

import type {
  Highlight,
  Logger,
  SearchMetrics,
  SearchReranker,
} from '@/tools/search/types';
import type { Reranker } from './types';
import { withDecisionDeadline } from '@/decisions/deadline';
import { BaseReranker } from '@/tools/search/rerankers';

/** Adapts index-based rankings to web-search highlights and its existing metrics/fallback policy. */
export function createWebSearchReranker(
  reranker: Reranker,
  options: { timeoutMs?: number; logger?: Logger } = {}
): SearchReranker {
  class SearchAdapter extends BaseReranker {
    readonly provider = reranker.id;

    async rerank(
      query: string,
      documents: string[],
      topK = 5,
      metrics?: SearchMetrics
    ): Promise<Highlight[]> {
      const candidates = [...documents];
      const run = this.beginRerank(candidates, topK, metrics);
      if (candidates.length === 0 || topK === 0) {
        return this.complete(run, []);
      }
      try {
        const response = await withDecisionDeadline(
          this.provider,
          options.timeoutMs ?? 10_000,
          undefined,
          (signal, waitFor) =>
            waitFor(
              reranker.rerank({
                query,
                documents: candidates,
                topK,
                signal,
                ...(options.timeoutMs !== undefined && {
                  timeoutMs: options.timeoutMs,
                }),
              })
            )
        );
        run.model = response.model;
        run.units = response.usage?.inputTokens;
        if (!Array.isArray(response.results) || response.results.length === 0) {
          return this.fallback(run, 'bad_response');
        }
        const seen = new Set<number>();
        const highlights: Highlight[] = [];
        let previousScore = Infinity;
        for (const result of response.results) {
          if (
            !Number.isSafeInteger(result.index) ||
            result.index < 0 ||
            result.index >= candidates.length ||
            seen.has(result.index) ||
            !Number.isFinite(result.score) ||
            result.score > previousScore
          ) {
            return this.fallback(run, 'invalid_results');
          }
          seen.add(result.index);
          previousScore = result.score;
          if (highlights.length < topK) {
            highlights.push({
              text: candidates[result.index],
              score: result.score,
            });
          }
        }
        return this.complete(run, highlights);
      } catch {
        return this.fallback(run, 'error', {
          message: 'reranker request failed',
        });
      }
    }
  }
  return new SearchAdapter(options.logger);
}

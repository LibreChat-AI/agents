/** What a cached score is keyed on: the document text, not an identity that survives edits. */
export interface RerankCacheKey {
  query: string;
  document: string;
  /** The model that produced the score; only an echo of the configured model is cached. */
  model: string;
  /** Identifies the rubric and instructions the score answers. */
  rubric: string;
}

/** Per-document scores; a host can supply its own (for example, hashed keys or a shared store). */
export interface RerankScoreCache {
  get(key: RerankCacheKey): number | undefined;
  set(key: RerankCacheKey, score: number): void;
}

export interface MemoryScoreCache extends RerankScoreCache {
  readonly size: number;
  clear(): void;
}

export interface ScoreCacheOptions {
  /** Entries kept, least recently used evicted first. Defaults to 20,000. */
  max?: number;
  /** Defaults to one day. */
  ttlMs?: number;
  now?: () => number;
}

/** A bounded in-memory LRU whose entries expire after `ttlMs`; keys hold the full document text. */
export function createScoreCache(
  options: ScoreCacheOptions = {}
): MemoryScoreCache {
  const max = Math.max(1, Math.floor(options.max ?? 20_000));
  const ttlMs = options.ttlMs ?? 86_400_000;
  const now = options.now ?? Date.now;
  const entries = new Map<string, { score: number; at: number }>();
  const keyOf = (key: RerankCacheKey): string =>
    JSON.stringify([key.query, key.document, key.model, key.rubric]);
  return {
    get(key: RerankCacheKey): number | undefined {
      const id = keyOf(key);
      const entry = entries.get(id);
      if (!entry) {
        return undefined;
      }
      entries.delete(id);
      if (now() - entry.at > ttlMs) {
        return undefined;
      }
      entries.set(id, entry);
      return entry.score;
    },
    set(key: RerankCacheKey, score: number): void {
      const id = keyOf(key);
      entries.delete(id);
      entries.set(id, { score, at: now() });
      while (entries.size > max) {
        const oldest = entries.keys().next();
        if (oldest.done === true) {
          break;
        }
        entries.delete(oldest.value);
      }
    },
    get size(): number {
      return entries.size;
    },
    clear(): void {
      entries.clear();
    },
  };
}

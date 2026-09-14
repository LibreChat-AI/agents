import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from './types';
import {
  WebSearchToolDescription,
  WebSearchToolName,
  countrySchema,
  imagesSchema,
  videosSchema,
  querySchema,
  dateSchema,
  newsSchema,
  DATE_RANGE,
} from './schema';
import { createSearchAPI, createSourceProcessor } from './search';
import { createKeenableScraper } from './keenable-scraper';
import { createSerperScraper } from './serper-scraper';
import { createTavilyScraper } from './tavily-scraper';
import { createFirecrawlScraper } from './firecrawl';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { createCrwScraper } from './crw-scraper';
import { expandHighlights } from './highlights';
import { createSearchMetrics } from './metrics';
import {
  formatResultsForLLM,
  MISSING_SEARCH_RESULT_DATA_ERROR,
} from './format';
import { createDefaultLogger } from './utils';
import { createReranker } from './rerankers';
import { Constants } from '@/common';

/**
 * Settled label for a `web_search` call's intent (see `intentArg.ts`).
 *
 * Counts the result kinds `formatResultsForLLM` actually renders —
 * `references` only tracks links embedded in extracted highlights, so it
 * undercounts ordinary results and can overcount when one highlight embeds
 * several links.
 *
 * A caught provider or processing failure is reported through `data.error`
 * while the tool still returns NORMALLY, so that case must author its own
 * label: the `ToolMessage` carries success status, so without an authored
 * outcome the in-flight intent ("Searching…") would stand as the settled
 * label and present a failed search as an ordinary one.
 *
 * Returns undefined for a genuine zero-result search, leaving the
 * model-authored intent to stand unchanged as the label.
 */
export function resolveSearchOutcome(
  data: t.SearchResultData,
  query: string
): string | undefined {
  if (data.error != null && data.error !== '') {
    return `Search failed for "${query}"`;
  }
  const count =
    (data.organic?.length ?? 0) +
    (data.topStories?.length ?? 0) +
    (data.news?.length ?? 0) +
    (data.images?.length ?? 0) +
    (data.videos?.length ?? 0) +
    (data.places?.length ?? 0) +
    (data.peopleAlsoAsk?.length ?? 0) +
    (data.knowledgeGraph != null ? 1 : 0) +
    (data.answerBox != null ? 1 : 0);
  if (count === 0) {
    return undefined;
  }
  return `Found ${count} result${count === 1 ? '' : 's'} for "${query}"`;
}

export function normalizeSearchResultData(
  result: t.SearchResultData | null | undefined
): t.SearchResultData {
  return result ?? { error: MISSING_SEARCH_RESULT_DATA_ERROR };
}

/** Distinct rows across the main search's two collections. SearXNG derives
 * both from one result array — a row matching its news heuristic lands in
 * `organic` and `topStories` alike — so summing the lengths would report
 * more rows than the provider actually returned. */
const countWebResults = (data: t.SearchResultData): number => {
  const organic = data.organic ?? [];
  const topStories = data.topStories ?? [];
  if (organic.length === 0 || topStories.length === 0) {
    return organic.length + topStories.length;
  }

  const links = new Set<string>();
  let unlinked = 0;
  for (const row of [...organic, ...topStories]) {
    if (row.link) {
      links.add(row.link);
      continue;
    }
    /** Nothing to dedupe a blank link against, so it counts on its own. */
    unlinked += 1;
  }
  return links.size + unlinked;
};

/** Rows a sub-search contributed, for the run summary's per-type breakdown. */
const countResults = (
  type: t.SubSearchType,
  data?: t.SearchResultData
): number => {
  if (data == null) {
    return 0;
  }
  if (type === 'images') {
    return data.images?.length ?? 0;
  }
  if (type === 'videos') {
    return data.videos?.length ?? 0;
  }
  if (type === 'news') {
    return data.news?.length ?? 0;
  }
  return countWebResults(data);
};

/**
 * Executes parallel searches and merges the results,
 * deduplicating top stories by link
 */
export async function executeParallelSearches({
  searchAPI,
  query,
  date,
  country,
  safeSearch,
  images,
  videos,
  news,
  logger,
  provider = 'unknown',
  metrics,
}: {
  searchAPI: ReturnType<typeof createSearchAPI>;
  query: string;
  date?: DATE_RANGE;
  country?: string;
  safeSearch: t.SearchToolConfig['safeSearch'];
  images: boolean;
  videos: boolean;
  news: boolean;
  logger: t.Logger;
  /** Labels the provider in the run summary. Optional so the pre-existing
   * call contract still holds for callers outside this package. */
  provider?: string;
  /** Collector owned by the caller. Without one, this call opens and flushes
   * its own, so a direct caller still gets the single summary line. */
  metrics?: t.SearchMetrics;
}): Promise<t.SearchResult> {
  const collector = metrics ?? createSearchMetrics(logger);
  /** A rejected main search is fatal, but rethrowing it from the task itself
   * would settle `Promise.all` while its siblings are still in flight, and
   * their observations would then land in an already-flushed phase. Every
   * task resolves; the failure is held here and raised once all have run. */
  let mainFailure: { error: unknown } | undefined;

  /** Sub-searches resolve rather than reject so their siblings still merge.
   * A rejected MAIN search rethrows the provider's own error below — callers
   * have always seen that error object, not a wrapped copy. */
  const runSearch = async (type: t.SubSearchType): Promise<t.SearchResult> => {
    const startedAt = Date.now();
    try {
      const result = await searchAPI.getSources({
        query,
        date,
        country,
        safeSearch,
        ...(type !== 'web' && { type }),
      });
      collector.recordSearch({
        provider,
        type,
        results: countResults(type, result.data),
        durationMs: Date.now() - startedAt,
        error: result.success ? undefined : (result.error ?? 'Search failed'),
      });
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      collector.recordSearch({
        provider,
        type,
        results: 0,
        durationMs: Date.now() - startedAt,
        error: message,
        thrown: true,
      });
      if (type === 'web') {
        mainFailure = { error };
      }
      return { success: false, error: message };
    }
  };

  // Prepare all search tasks to run in parallel
  const searchTasks: Promise<t.SearchResult>[] = [runSearch('web')];

  if (images) {
    searchTasks.push(runSearch('images'));
  }
  if (videos) {
    searchTasks.push(runSearch('videos'));
  }
  if (news) {
    searchTasks.push(runSearch('news'));
  }

  // Run all searches in parallel. No task rejects, so every observation is
  // recorded before the collector is flushed or a failure is raised.
  const results = await Promise.all(searchTasks);
  if (metrics == null) {
    collector.flush();
  }

  if (mainFailure != null) {
    throw mainFailure.error;
  }

  // Get the main search result (first result)
  const mainResult = results[0];
  if (!mainResult.success) {
    throw new Error(mainResult.error ?? 'Search failed');
  }

  // Merge additional results with the main results
  const mergedResults = { ...mainResult.data };

  // Convert existing news to topStories if present
  if (mergedResults.news !== undefined && mergedResults.news.length > 0) {
    const existingNewsAsTopStories = mergedResults.news
      .filter((newsItem) => newsItem.link !== undefined && newsItem.link !== '')
      .map((newsItem) => ({
        title: newsItem.title ?? '',
        link: newsItem.link ?? '',
        source: newsItem.source ?? '',
        date: newsItem.date ?? '',
        imageUrl: newsItem.imageUrl ?? '',
        processed: false,
      }));
    mergedResults.topStories = [
      ...(mergedResults.topStories ?? []),
      ...existingNewsAsTopStories,
    ];
    delete mergedResults.news;
  }

  results.slice(1).forEach((result) => {
    if (result.success && result.data !== undefined) {
      if (result.data.images !== undefined && result.data.images.length > 0) {
        mergedResults.images = [
          ...(mergedResults.images ?? []),
          ...result.data.images,
        ];
      }
      if (result.data.videos !== undefined && result.data.videos.length > 0) {
        mergedResults.videos = [
          ...(mergedResults.videos ?? []),
          ...result.data.videos,
        ];
      }
      if (result.data.news !== undefined && result.data.news.length > 0) {
        const newsAsTopStories = result.data.news.map((newsItem) => ({
          ...newsItem,
          link: newsItem.link ?? '',
        }));
        mergedResults.topStories = [
          ...(mergedResults.topStories ?? []),
          ...newsAsTopStories,
        ];
      }
    }
  });

  if (
    mergedResults.topStories !== undefined &&
    mergedResults.topStories.length > 1
  ) {
    /** The main search's own news results and the parallel news sub-search
     * frequently return the same stories — keep the first occurrence of each
     * link so duplicates aren't scraped, reranked, and formatted repeatedly */
    const seenLinks = new Set<string>();
    mergedResults.topStories = mergedResults.topStories.filter((story) => {
      if (!story.link || seenLinks.has(story.link)) {
        return false;
      }
      seenLinks.add(story.link);
      return true;
    });
  }

  return { success: true, data: mergedResults };
}

function createSearchProcessor({
  searchAPI,
  provider,
  safeSearch,
  supportsImages,
  supportsVideos,
  supportsNews,
  sourceProcessor,
  onGetHighlights,
  mainExpandBy,
  separatorExpandBy,
  logger,
}: {
  provider: string;
  safeSearch: t.SearchToolConfig['safeSearch'];
  supportsImages: boolean;
  supportsVideos: boolean;
  supportsNews: boolean;
  searchAPI: ReturnType<typeof createSearchAPI>;
  sourceProcessor: ReturnType<typeof createSourceProcessor>;
  onGetHighlights: t.SearchToolConfig['onGetHighlights'];
  mainExpandBy: t.SearchToolConfig['mainExpandBy'];
  separatorExpandBy: t.SearchToolConfig['separatorExpandBy'];
  logger: t.Logger;
}) {
  return async function ({
    query,
    date,
    country,
    proMode = true,
    maxSources = 5,
    onSearchResults,
    images = false,
    videos = false,
    news = false,
  }: {
    query: string;
    country?: string;
    date?: DATE_RANGE;
    proMode?: boolean;
    maxSources?: number;
    onSearchResults: t.SearchToolConfig['onSearchResults'];
    images?: boolean;
    videos?: boolean;
    news?: boolean;
  }): Promise<t.SearchResultData> {
    /** One collector for the whole call: the provider, scrape, and rerank
     * phases each fold into counters and flush together, so a search costs a
     * bounded handful of lines instead of a few per source. */
    const metrics = createSearchMetrics(logger);
    try {
      // Execute parallel searches and merge results
      const searchResult = await executeParallelSearches({
        searchAPI,
        query,
        date,
        country,
        safeSearch,
        images: supportsImages && images,
        videos: supportsVideos && videos,
        news: supportsNews && news,
        logger,
        provider,
        metrics,
      });

      onSearchResults?.(searchResult);

      const processedSources = await sourceProcessor.processSources({
        query,
        news,
        metrics,
        result: searchResult,
        proMode,
        onGetHighlights,
        numElements: maxSources,
      });

      return expandHighlights(
        processedSources,
        mainExpandBy,
        separatorExpandBy
      );
    } catch (error) {
      logger.error('Error in search:', error);
      return {
        organic: [],
        topStories: [],
        images: [],
        videos: [],
        news: [],
        relatedSearches: [],
        error: error instanceof Error ? error.message : String(error),
      };
    } finally {
      metrics.flush();
    }
  };
}

function createOnSearchResults({
  runnableConfig,
  onSearchResults,
}: {
  runnableConfig: RunnableConfig;
  onSearchResults: t.SearchToolConfig['onSearchResults'];
}) {
  return function (results: t.SearchResult): void {
    if (!onSearchResults) {
      return;
    }
    onSearchResults(results, runnableConfig);
  };
}

function createTool({
  schema,
  search,
  maxOutputChars,
  onSearchResults: _onSearchResults,
}: {
  schema: Record<string, unknown>;
  search: ReturnType<typeof createSearchProcessor>;
  maxOutputChars?: number;
  onSearchResults: t.SearchToolConfig['onSearchResults'];
}): DynamicStructuredTool {
  return tool(
    async (rawParams, runnableConfig) => {
      const params = rawParams as SearchToolParams;
      const { query, date, country: _c, images, videos, news } = params;
      const country = typeof _c === 'string' && _c ? _c : undefined;
      const searchResult = await search({
        query,
        date,
        country,
        images,
        videos,
        news,
        onSearchResults: createOnSearchResults({
          runnableConfig,
          onSearchResults: _onSearchResults,
        }),
      });
      const turn = runnableConfig.toolCall?.turn ?? 0;
      const resultData = normalizeSearchResultData(searchResult);
      const { output, references } = formatResultsForLLM(
        turn,
        resultData,
        maxOutputChars
      );
      const data: t.SearchResultData = { turn, ...resultData, references };
      const outcome = resolveSearchOutcome(data, query);
      return [
        output,
        { [Constants.WEB_SEARCH]: data, ...(outcome != null && { outcome }) },
      ];
    },
    {
      name: WebSearchToolName,
      description: WebSearchToolDescription,
      schema: schema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

/**
 * Creates a search tool with configurable search and scraper providers.
 *
 * Search providers: Serper (Google results), SearXNG (self-hosted meta-search), Tavily (AI-optimized), fastCRW (Firecrawl-compatible, self-host or cloud).
 * Scraper providers: Firecrawl (default, full-featured), Serper (lightweight), Tavily (batch extraction), fastCRW (Firecrawl-compatible, self-host or cloud).
 *
 * The country schema field is exposed to the LLM for providers that support localized results.
 */
/** Input params type for search tool */
interface SearchToolParams {
  query: string;
  date?: DATE_RANGE;
  country?: string;
  images?: boolean;
  videos?: boolean;
  news?: boolean;
}

export const createSearchTool = (
  config: t.SearchToolConfig = {}
): DynamicStructuredTool => {
  const {
    searchProvider = 'serper',
    serperApiKey,
    searxngInstanceUrl,
    searxngApiKey,
    searxngSearchOptions,
    tavilyApiKey,
    tavilySearchUrl,
    tavilyExtractUrl,
    tavilySearchOptions,
    keenableApiKey,
    keenableApiUrl,
    keenableSearchOptions,
    keenableScraperOptions,
    rerankerType = 'cohere',
    rerankerTimeout,
    topResults = 5,
    maxContentLength,
    chunkSize,
    chunkOverlap,
    mainExpandBy,
    separatorExpandBy,
    maxOutputChars,
    strategies = ['no_extraction'],
    filterContent = true,
    safeSearch = 1,
    scraperProvider = 'firecrawl',
    firecrawlApiKey,
    firecrawlApiUrl,
    firecrawlVersion,
    firecrawlOptions,
    serperScraperOptions,
    tavilyScraperOptions,
    crwApiKey,
    crwApiUrl,
    crwSearchOptions,
    crwScraperOptions,
    scraperTimeout,
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
    httpAgent,
    httpsAgent,
    onSearchResults: _onSearchResults,
    onGetHighlights,
  } = config;

  const logger = config.logger || createDefaultLogger();
  const effectiveTavilySearchOptions =
    searchProvider === 'tavily' && config.safeSearch != null
      ? {
        ...tavilySearchOptions,
        safeSearch: config.safeSearch !== 0,
      }
      : tavilySearchOptions;

  const schemaProperties: Record<string, unknown> = {
    intent: { ...INTENT_PROPERTY },
    query: querySchema,
    date: dateSchema,
    images: imagesSchema,
    videos: videosSchema,
    news: newsSchema,
  };

  if (searchProvider === 'serper' || searchProvider === 'tavily') {
    schemaProperties.country = countrySchema;
  }

  const toolSchema = {
    type: 'object',
    properties: schemaProperties,
    required: ['query'],
  };

  const searchAPI = createSearchAPI({
    searchProvider,
    serperApiKey,
    searxngInstanceUrl,
    searxngApiKey,
    searxngSearchOptions,
    tavilyApiKey,
    tavilySearchUrl,
    tavilySearchOptions: effectiveTavilySearchOptions,
    keenableApiKey,
    keenableApiUrl,
    keenableSearchOptions,
    crwApiKey,
    crwApiUrl,
    crwSearchOptions,
    httpAgent,
    httpsAgent,
  });

  /** Create scraper based on scraperProvider */
  let scraperInstance: t.BaseScraper;

  if (scraperProvider === 'serper') {
    scraperInstance = createSerperScraper({
      ...serperScraperOptions,
      apiKey: serperApiKey,
      timeout: scraperTimeout ?? serperScraperOptions?.timeout,
      httpAgent: httpAgent ?? serperScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? serperScraperOptions?.httpsAgent,
      logger,
    });
  } else if (scraperProvider === 'tavily') {
    scraperInstance = createTavilyScraper({
      ...tavilyScraperOptions,
      apiKey:
        tavilyScraperOptions?.apiKey ??
        tavilyApiKey ??
        process.env.TAVILY_API_KEY,
      apiUrl: tavilyScraperOptions?.apiUrl ?? tavilyExtractUrl,
      timeout: scraperTimeout ?? tavilyScraperOptions?.timeout,
      httpAgent: httpAgent ?? tavilyScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? tavilyScraperOptions?.httpsAgent,
      logger,
    });
  } else if (scraperProvider === 'crw') {
    scraperInstance = createCrwScraper({
      ...crwScraperOptions,
      apiKey: crwScraperOptions?.apiKey ?? crwApiKey ?? process.env.CRW_API_KEY,
      apiUrl: crwScraperOptions?.apiUrl ?? crwApiUrl,
      timeout: scraperTimeout ?? crwScraperOptions?.timeout,
      formats: crwScraperOptions?.formats ?? ['markdown', 'rawHtml'],
      httpAgent: httpAgent ?? crwScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? crwScraperOptions?.httpsAgent,
      logger,
    });
  } else if (scraperProvider === 'keenable') {
    scraperInstance = createKeenableScraper({
      ...keenableScraperOptions,
      apiKey: keenableScraperOptions?.apiKey ?? keenableApiKey,
      timeout: scraperTimeout ?? keenableScraperOptions?.timeout,
      attributionTitle:
        keenableScraperOptions?.attributionTitle ??
        keenableSearchOptions?.attributionTitle,
      httpAgent: httpAgent ?? keenableScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? keenableScraperOptions?.httpsAgent,
      logger,
    });
  } else {
    scraperInstance = createFirecrawlScraper({
      ...firecrawlOptions,
      apiKey: firecrawlApiKey ?? process.env.FIRECRAWL_API_KEY,
      apiUrl: firecrawlApiUrl,
      version: firecrawlVersion,
      timeout: scraperTimeout ?? firecrawlOptions?.timeout,
      formats: firecrawlOptions?.formats ?? ['markdown', 'rawHtml'],
      httpAgent: httpAgent ?? firecrawlOptions?.httpAgent,
      httpsAgent: httpsAgent ?? firecrawlOptions?.httpsAgent,
      logger,
    });
  }

  const selectedReranker = createReranker({
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
    httpAgent,
    httpsAgent,
    logger,
  });

  /** `none` is a deliberate opt-out that `createReranker` already reports;
   * only an unusable configuration warrants a warning here. */
  if (!selectedReranker && rerankerType !== 'none') {
    logger.warn('No reranker selected. Using default ranking.');
  }

  const sourceProcessor = createSourceProcessor(
    {
      reranker: selectedReranker,
      topResults,
      maxContentLength,
      chunkSize,
      chunkOverlap,
      strategies,
      filterContent,
      logger,
    },
    scraperInstance
  );

  const search = createSearchProcessor({
    searchAPI,
    provider: searchProvider,
    safeSearch,
    // Keenable is organic-only: its API ignores `type`, so image/news
    // sub-searches would spend rate limit and merge nothing.
    supportsImages: searchProvider !== 'keenable',
    supportsVideos:
      searchProvider !== 'tavily' &&
      searchProvider !== 'keenable' &&
      searchProvider !== 'crw',
    supportsNews: searchProvider !== 'keenable',
    sourceProcessor,
    onGetHighlights,
    mainExpandBy,
    separatorExpandBy,
    logger,
  });

  return createTool({
    search,
    schema: toolSchema,
    maxOutputChars,
    onSearchResults: _onSearchResults,
  });
};

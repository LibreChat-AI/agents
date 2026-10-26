import axios from 'axios';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import type * as t from './types';
import {
  getAttribution,
  createDefaultLogger,
  formatErrorForLog,
} from './utils';
import { createKeenableAPI } from './keenable-search';
import { createTavilyAPI } from './tavily-search';
import { createSearchMetrics } from './metrics';
import { getDefaultRanking } from './rerankers';
import { createCrwAPI } from './crw-search';

/** Engines queried when `searxngSearchOptions.engines` is not configured. */
const DEFAULT_SEARXNG_ENGINES = 'google,bing,duckduckgo';

const chunker = {
  cleanText: (text: string): string => {
    if (!text) return '';

    /** Normalized all line endings to '\n' */
    const normalizedText = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    /** Handle multiple backslashes followed by newlines
     * This replaces patterns like '\\\\\\n' with a single newline */
    const fixedBackslashes = normalizedText.replace(/\\+\n/g, '\n');

    /** Cleaned up consecutive newlines, tabs, and spaces around newlines */
    const cleanedNewlines = fixedBackslashes.replace(/[\t ]*\n[\t \n]*/g, '\n');

    /** Cleaned up excessive spaces and tabs */
    const cleanedSpaces = cleanedNewlines.replace(/[ \t]+/g, ' ');

    return cleanedSpaces.trim();
  },
  splitText: async (
    text: string,
    options?: {
      chunkSize?: number;
      chunkOverlap?: number;
      separators?: string[];
    }
  ): Promise<string[]> => {
    const chunkSize = options?.chunkSize ?? 150;
    const chunkOverlap = options?.chunkOverlap ?? 50;
    const separators = options?.separators || ['\n\n', '\n'];

    const splitter = new RecursiveCharacterTextSplitter({
      separators,
      chunkSize,
      chunkOverlap,
    });

    return await splitter.splitText(text);
  },

  splitTexts: async (
    texts: string[],
    options?: {
      chunkSize?: number;
      chunkOverlap?: number;
      separators?: string[];
    },
    logger?: t.Logger
  ): Promise<string[][]> => {
    // Split multiple texts
    const logger_ = logger || createDefaultLogger();
    const promises = texts.map((text) =>
      chunker.splitText(text, options).catch((error) => {
        logger_.error('Error splitting text:', error);
        return [text];
      })
    );
    return Promise.all(promises);
  },
};

const DEFAULT_MAX_CONTENT_LENGTH = 50000;
const DEFAULT_CHUNK_SIZE = 150;
const DEFAULT_CHUNK_OVERLAP = 50;

/** Resolves reranker chunking from config, the `SEARCH_CHUNK_SIZE` /
 * `SEARCH_CHUNK_OVERLAP` env vars, or the defaults (150 / 50 chars). The
 * overlap is clamped below the chunk size — `RecursiveCharacterTextSplitter`
 * throws when overlap >= size. */
function resolveChunkOptions(
  chunkSize?: number,
  chunkOverlap?: number
): { chunkSize: number; chunkOverlap: number } {
  const resolve = (
    configValue: number | undefined,
    envVar: string,
    fallback: number
  ): number => {
    if (configValue != null && configValue > 0) {
      return configValue;
    }
    const envValue = Number(process.env[envVar]);
    if (Number.isFinite(envValue) && envValue > 0) {
      return envValue;
    }
    return fallback;
  };

  const size = resolve(chunkSize, 'SEARCH_CHUNK_SIZE', DEFAULT_CHUNK_SIZE);
  let overlap = resolve(
    chunkOverlap,
    'SEARCH_CHUNK_OVERLAP',
    DEFAULT_CHUNK_OVERLAP
  );
  if (overlap >= size) {
    overlap = Math.floor(size / 3);
  }
  return { chunkSize: size, chunkOverlap: overlap };
}

/** Resolves the per-source scraped content cap from config, the
 * `SEARCH_MAX_CONTENT_LENGTH` env var, or the default (50,000 chars) */
function resolveMaxContentLength(maxContentLength?: number): number {
  if (maxContentLength != null && maxContentLength > 0) {
    return maxContentLength;
  }
  const envValue = Number(process.env.SEARCH_MAX_CONTENT_LENGTH);
  if (Number.isFinite(envValue) && envValue > 0) {
    return envValue;
  }
  return DEFAULT_MAX_CONTENT_LENGTH;
}

function truncateContent(content: string, maxLength: number): string {
  return content.length > maxLength ? content.slice(0, maxLength) : content;
}

function createSourceUpdateCallback(sourceMap: Map<string, t.ValidSource>) {
  return (link: string, update?: Partial<t.ValidSource>): void => {
    const source = sourceMap.get(link);
    if (source) {
      sourceMap.set(link, {
        ...source,
        ...update,
      });
    }
  };
}

/** Provider label for the metrics summary when reranking is switched off.
 * `undefined` would be indistinguishable from a reranker that never ran. */
const NO_RERANKER_PROVIDER = 'none';

/** Returns undefined without logging when there is nothing to rank: an empty
 * scrape is already counted by the scrape summary.
 *
 * A *missing* reranker is not the same as nothing to rank. `rerankerType:
 * 'none'` is a schema-valid opt-out, and `createReranker` returns `undefined`
 * for it by design. Returning `undefined` here as well would leave the source
 * without highlights, and `expandHighlights` then strips its content — the
 * search would answer with links and no text at all. So the chunks are handed
 * on in their original order via `getDefaultRanking`, exactly what every
 * reranker's own failure path already does. */
const getHighlights = async ({
  query,
  content,
  reranker,
  metrics,
  topResults = 5,
  maxContentLength = DEFAULT_MAX_CONTENT_LENGTH,
  chunkOptions,
}: {
  content: string;
  query: string;
  reranker?: t.SearchReranker;
  metrics: t.SearchMetrics;
  topResults?: number;
  maxContentLength?: number;
  chunkOptions?: { chunkSize: number; chunkOverlap: number };
}): Promise<t.Highlight[] | undefined> => {
  if (!content) {
    return;
  }

  /** Both failures are this reranker's attempt for this source, so they stay
   * in its phase — but they are caught separately: sharing one `catch` would
   * report a reranker that threw as a chunking failure, with a chunk count
   * of zero for text that split fine. */
  const chunkStartedAt = Date.now();
  let documents: string[];
  try {
    documents = await chunker.splitText(
      truncateContent(content, maxContentLength),
      chunkOptions
    );
  } catch (error) {
    metrics.recordRerank({
      provider: reranker?.provider ?? NO_RERANKER_PROVIDER,
      chunks: 0,
      results: 0,
      durationMs: Date.now() - chunkStartedAt,
      reason: 'chunk_error',
      error: formatErrorForLog(error),
    });
    return;
  }

  /** Reranking is switched off, so the chunks pass through unscored. Recorded
   * like any other rerank so the search summary still accounts for the source;
   * it carries no `reason`, because nothing failed. */
  if (!reranker) {
    const highlights = getDefaultRanking(documents, topResults);
    metrics.recordRerank({
      provider: NO_RERANKER_PROVIDER,
      chunks: documents.length,
      results: highlights.length,
      durationMs: Date.now() - chunkStartedAt,
    });
    return highlights;
  }

  const rerankStartedAt = Date.now();
  try {
    return await reranker.rerank(query, documents, topResults, metrics);
  } catch (error) {
    /** The bundled rerankers absorb their own failures and record the
     * observation themselves; a custom implementation may reject instead. */
    metrics.recordRerank({
      provider: reranker.provider,
      chunks: documents.length,
      results: 0,
      durationMs: Date.now() - rerankStartedAt,
      reason: 'error',
      error: formatErrorForLog(error),
    });
    return;
  }
};

const createSerperAPI = (
  apiKey?: string,
  agents?: t.HttpAgentConfig
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const baseUrl = process.env.SERPER_BASE_URL;
  const config = {
    apiKey: apiKey ?? process.env.SERPER_API_KEY,
    baseUrl: (baseUrl == null || baseUrl === ''
      ? 'https://google.serper.dev'
      : baseUrl
    ).replace(/\/+$/, ''),
    timeout: 10000,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('SERPER_API_KEY is required for SerperAPI');
  }

  const getSources = async ({
    query,
    date,
    country,
    safeSearch,
    numResults = 8,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const safe = ['off', 'moderate', 'active'] as const;
      const payload: t.SerperSearchPayload = {
        q: query,
        safe: safe[safeSearch ?? 1],
        num: Math.min(Math.max(1, numResults), 10),
      };

      // Set the search type if provided
      if (type) {
        payload.type = type;
      }

      if (date != null) {
        payload.tbs = `qdr:${date}`;
      }

      if (country != null && country !== '') {
        payload['gl'] = country.toLowerCase();
      }

      // Determine the API endpoint based on the search type
      let apiEndpoint = `${config.baseUrl}/search`;
      if (type === 'images') {
        apiEndpoint = `${config.baseUrl}/images`;
      } else if (type === 'videos') {
        apiEndpoint = `${config.baseUrl}/videos`;
      } else if (type === 'news') {
        apiEndpoint = `${config.baseUrl}/news`;
      }

      const response = await axios.post<t.SerperResultData>(
        apiEndpoint,
        payload,
        {
          headers: {
            'X-API-KEY': config.apiKey,
            'Content-Type': 'application/json',
          },
          timeout: config.timeout,
          httpAgent: agents?.httpAgent,
          httpsAgent: agents?.httpsAgent,
        }
      );

      const data = response.data;
      const results: t.SearchResultData = {
        organic: data.organic,
        images: data.images ?? [],
        answerBox: data.answerBox,
        topStories: data.topStories ?? [],
        peopleAlsoAsk: data.peopleAlsoAsk,
        knowledgeGraph: data.knowledgeGraph,
        relatedSearches: data.relatedSearches,
        videos: data.videos ?? [],
        news: data.news ?? [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return { success: false, error: `API request failed: ${errorMessage}` };
    }
  };

  return { getSources };
};

const createSearXNGAPI = (
  instanceUrl?: string,
  apiKey?: string,
  options?: t.SearxNGSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const engines = options?.engines?.trim();
  const language = options?.language?.trim();
  const config = {
    instanceUrl: instanceUrl ?? process.env.SEARXNG_INSTANCE_URL,
    apiKey: apiKey ?? process.env.SEARXNG_API_KEY,
    engines:
      engines != null && engines !== '' ? engines : DEFAULT_SEARXNG_ENGINES,
    language: language != null && language !== '' ? language : 'all',
    timeRange: options?.timeRange,
    timeout: options?.timeout ?? 10000,
  };

  if (config.instanceUrl == null || config.instanceUrl === '') {
    throw new Error('SEARXNG_INSTANCE_URL is required for SearXNG API');
  }

  const getSources = async ({
    query,
    numResults = 8,
    safeSearch,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      // Ensure the instance URL ends with /search
      if (config.instanceUrl == null || config.instanceUrl === '') {
        return { success: false, error: 'Instance URL is not defined' };
      }

      let searchUrl = config.instanceUrl;
      if (!searchUrl.endsWith('/search')) {
        searchUrl = searchUrl.replace(/\/$/, '') + '/search';
      }

      // Determine the search category based on the type
      let category = 'general';
      if (type === 'images') {
        category = 'images';
      } else if (type === 'videos') {
        category = 'videos';
      } else if (type === 'news') {
        category = 'news';
      }

      // Prepare parameters for SearXNG
      const params: t.SearxNGSearchPayload = {
        q: query,
        format: 'json',
        pageno: 1,
        categories: category,
        language: config.language,
        safesearch: safeSearch,
        engines: config.engines,
      };

      if (config.timeRange != null) {
        params.time_range = config.timeRange;
      }

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (config.apiKey != null && config.apiKey !== '') {
        headers['X-API-Key'] = config.apiKey;
      }

      const response = await axios.get(searchUrl, {
        headers,
        params,
        timeout: config.timeout,
        httpAgent: options?.httpAgent,
        httpsAgent: options?.httpsAgent,
      });

      const data = response.data;

      // Helper function to identify news results since SearXNG doesn't provide that classification by default
      const isNewsResult = (result: t.SearXNGResult): boolean => {
        const url = result.url?.toLowerCase() ?? '';
        const title = result.title?.toLowerCase() ?? '';

        // News-related keywords in title/content
        const newsKeywords = [
          'breaking news',
          'latest news',
          'top stories',
          'news today',
          'developing story',
          'trending news',
          'news',
        ];

        // Check if title/content contains news keywords
        const hasNewsKeywords = newsKeywords.some(
          (keyword) => title.toLowerCase().includes(keyword) // just title probably fine, content parsing is overkill for what we need: || content.includes(keyword)
        );

        // Check if URL contains news-related paths
        const hasNewsPath =
          url.includes('/news/') ||
          url.includes('/world/') ||
          url.includes('/politics/') ||
          url.includes('/breaking/');

        return hasNewsKeywords || hasNewsPath;
      };

      // Transform SearXNG results to match SerperAPI format
      const organicResults = (data.results ?? [])
        .slice(0, numResults)
        .map((result: t.SearXNGResult, index: number) => {
          let attribution: string;
          try {
            attribution = new URL(result.url ?? '').hostname;
          } catch {
            attribution = '';
          }

          return {
            position: index + 1,
            title: result.title ?? '',
            link: result.url ?? '',
            snippet: result.content ?? '',
            date: result.publishedDate ?? '',
            attribution,
          };
        });

      const imageResults = (data.results ?? [])
        .filter((result: t.SearXNGResult) => result.img_src)
        .slice(0, 6)
        .map((result: t.SearXNGResult, index: number) => ({
          title: result.title ?? '',
          imageUrl: result.img_src ?? '',
          position: index + 1,
          source: new URL(result.url ?? '').hostname,
          domain: new URL(result.url ?? '').hostname,
          link: result.url ?? '',
        }));

      // Extract news results from organic results
      const newsResults = (data.results ?? [])
        .filter(isNewsResult)
        .map((result: t.SearXNGResult, index: number) => {
          let attribution: string;
          try {
            attribution = new URL(result.url ?? '').hostname;
          } catch {
            attribution = '';
          }

          return {
            title: result.title ?? '',
            link: result.url ?? '',
            snippet: result.content ?? '',
            date: result.publishedDate ?? '',
            source: attribution,
            imageUrl: result.img_src ?? '',
            position: index + 1,
          };
        });

      const topStories = newsResults.slice(0, 5);

      const relatedSearches = Array.isArray(data.suggestions)
        ? data.suggestions.map((suggestion: string) => ({ query: suggestion }))
        : [];

      const results: t.SearchResultData = {
        organic: organicResults,
        images: imageResults,
        topStories: topStories, // Use first 5 extracted news as top stories
        relatedSearches,
        videos: [],
        news: newsResults,
        // Add empty arrays for other Serper fields to maintain parity
        places: [],
        shopping: [],
        peopleAlsoAsk: [],
        knowledgeGraph: undefined,
        answerBox: undefined,
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `SearXNG API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};

export const createSearchAPI = (
  config: t.SearchConfig
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const {
    searchProvider = 'serper',
    serperApiKey,
    searxngInstanceUrl,
    searxngApiKey,
    searxngSearchOptions,
    tavilyApiKey,
    tavilySearchUrl,
    tavilySearchOptions,
    keenableApiKey,
    keenableApiUrl,
    keenableSearchOptions,
    crwApiKey,
    crwApiUrl,
    crwSearchOptions,
    httpAgent,
    httpsAgent,
  } = config;

  const agents: t.HttpAgentConfig = { httpAgent, httpsAgent };

  if (searchProvider.toLowerCase() === 'serper') {
    return createSerperAPI(serperApiKey, agents);
  } else if (searchProvider.toLowerCase() === 'searxng') {
    return createSearXNGAPI(searxngInstanceUrl, searxngApiKey, {
      ...searxngSearchOptions,
      httpAgent: httpAgent ?? searxngSearchOptions?.httpAgent,
      httpsAgent: httpsAgent ?? searxngSearchOptions?.httpsAgent,
    });
  } else if (searchProvider.toLowerCase() === 'tavily') {
    return createTavilyAPI(tavilyApiKey, tavilySearchUrl, {
      ...tavilySearchOptions,
      httpAgent: httpAgent ?? tavilySearchOptions?.httpAgent,
      httpsAgent: httpsAgent ?? tavilySearchOptions?.httpsAgent,
    });
  } else if (searchProvider.toLowerCase() === 'keenable') {
    return createKeenableAPI(keenableApiKey, keenableApiUrl, {
      ...keenableSearchOptions,
      httpAgent: httpAgent ?? keenableSearchOptions?.httpAgent,
      httpsAgent: httpsAgent ?? keenableSearchOptions?.httpsAgent,
    });
  } else if (searchProvider.toLowerCase() === 'crw') {
    return createCrwAPI(crwApiKey, crwApiUrl, {
      ...crwSearchOptions,
      httpAgent: httpAgent ?? crwSearchOptions?.httpAgent,
      httpsAgent: httpsAgent ?? crwSearchOptions?.httpsAgent,
    });
  } else {
    throw new Error(
      `Invalid search provider: ${searchProvider}. Must be 'serper', 'searxng', 'tavily', 'keenable', or 'crw'`
    );
  }
};

export const createSourceProcessor = (
  config: t.ProcessSourcesConfig = {},
  scraperInstance?: t.BaseScraper
): {
  processSources: (
    fields: t.ProcessSourcesFields
  ) => Promise<t.SearchResultData>;
  topResults: number;
} => {
  if (!scraperInstance) {
    throw new Error('Scraper instance is required');
  }
  const {
    topResults = 5,
    // strategies = ['no_extraction'],
    // filterContent = true,
    reranker,
    logger,
  } = config;

  const maxContentLength = resolveMaxContentLength(config.maxContentLength);
  const chunkOptions = resolveChunkOptions(
    config.chunkSize,
    config.chunkOverlap
  );
  const logger_ = logger || createDefaultLogger();
  const scraper = scraperInstance;

  const processResponse = (
    url: string,
    response: t.AnyScraperResponse
  ): t.ScrapeResult => {
    const rawMetadata = scraper.extractMetadata(response);
    const metadata =
      Object.keys(rawMetadata).length > 0 ? rawMetadata : undefined;
    const attribution = getAttribution(url, metadata, logger_);

    if (response.success && response.data) {
      const [content, references] = scraper.extractContent(response);
      return {
        url,
        references,
        attribution,
        content: truncateContent(chunker.cleanText(content), maxContentLength),
      };
    }

    return { url, attribution, error: true, content: '' };
  };

  const addHighlights = async (
    result: t.ScrapeResult,
    query: string,
    metrics: t.SearchMetrics,
    onGetHighlights: t.SearchToolConfig['onGetHighlights']
  ): Promise<t.ScrapeResult> => {
    const highlights = await getHighlights({
      query,
      reranker,
      metrics,
      topResults,
      content: result.content,
      maxContentLength,
      chunkOptions,
    });
    if (onGetHighlights) {
      onGetHighlights(result.url);
    }
    return { ...result, highlights };
  };

  /** Scrape and rerank one link, recording the single observation that link
   * contributes to the run summary — the only place a per-link outcome is
   * reported, so nothing here logs per link. */
  const processLink = async (
    url: string,
    response: t.AnyScraperResponse,
    query: string,
    metrics: t.SearchMetrics,
    onGetHighlights: t.SearchToolConfig['onGetHighlights']
  ): Promise<t.ScrapeResult> => {
    /** `extractContent`/`extractMetadata` are the scraper implementation's
     * code: one malformed response must not reject alongside its siblings,
     * which would discard their results and their observations with them. */
    let scraped: t.ScrapeResult;
    try {
      scraped = processResponse(url, response);
    } catch (error) {
      metrics.recordScrape({ url, error: String(error) });
      return { url, error: true, content: '' };
    }
    if (scraped.error === true) {
      metrics.recordScrape({ url, error: response.error ?? 'Unknown error' });
      return scraped;
    }
    /** `getHighlights` absorbs its own failures, so only a throwing
     * `onGetHighlights` consumer reaches here; one bad callback must not
     * discard the sibling links awaiting alongside it. */
    const result = await addHighlights(
      scraped,
      query,
      metrics,
      onGetHighlights
    ).catch((error) => {
      logger_.error('Error processing scraped content:', error);
      return scraped;
    });
    metrics.recordScrape({
      url,
      chars: result.content.length,
      highlights: result.highlights?.length ?? 0,
    });
    return result;
  };

  const webScraper = {
    scrapeMany: async ({
      query,
      links,
      metrics,
      onGetHighlights,
    }: {
      query: string;
      links: string[];
      metrics: t.SearchMetrics;
      onGetHighlights: t.SearchToolConfig['onGetHighlights'];
    }): Promise<Array<t.ScrapeResult>> => {
      let responses: Array<[string, t.AnyScraperResponse]>;

      /** Scoped to acquisition alone. A batch `scrapeUrls` that rejects
       * yields no per-link responses, so nothing downstream will ever report
       * these links — without recording them here a total outage would flush
       * no scrape summary at all, reading exactly like a search that never
       * scraped anything. */
      try {
        if (scraper.scrapeUrls) {
          responses = await scraper.scrapeUrls(links);
        } else {
          responses = await Promise.all(
            links.map((link) =>
              scraper
                .scrapeUrl(link, {})
                .catch((error): [string, t.AnyScraperResponse] => [
                  link,
                  { success: false, error: String(error) },
                ])
            )
          );
        }
      } catch (error) {
        logger_.error('Error in scrapeMany:', error);
        const message = String(error);
        for (const link of links) {
          metrics.recordScrape({ url: link, error: message });
        }
        return [];
      }

      try {
        return await Promise.all(
          responses.map(([url, response]) =>
            processLink(url, response, query, metrics, onGetHighlights)
          )
        );
      } catch (error) {
        /** `processLink` absorbs its own failures and has already recorded
         * whatever it reached, so this only preserves the soft failure the
         * caller has always seen — it must not re-count these links. */
        logger_.error('Error in scrapeMany:', error);
        return [];
      }
    },
  };

  const fetchContents = async ({
    links,
    query,
    target,
    metrics,
    onGetHighlights,
    onContentScraped,
  }: {
    links: string[];
    query: string;
    target: number;
    metrics: t.SearchMetrics;
    onGetHighlights: t.SearchToolConfig['onGetHighlights'];
    onContentScraped?: (link: string, update?: Partial<t.ValidSource>) => void;
  }): Promise<void> => {
    const initialLinks = links.slice(0, target);
    // const remainingLinks = links.slice(target).reverse();
    const results = await webScraper.scrapeMany({
      query,
      metrics,
      links: initialLinks,
      onGetHighlights,
    });
    for (const result of results) {
      if (result.error === true) {
        continue;
      }
      const { url, content, attribution, references, highlights } = result;
      onContentScraped?.(url, {
        content,
        attribution,
        references,
        highlights,
      });
    }
  };

  const processSources = async ({
    result,
    numElements,
    query,
    news,
    proMode = true,
    onGetHighlights,
    metrics: ownerMetrics,
  }: t.ProcessSourcesFields): Promise<t.SearchResultData> => {
    /** The caller owns the collector when it supplies one — it has phases of
     * its own to fold in and flushes them together. */
    const metrics = ownerMetrics ?? createSearchMetrics(logger_);
    try {
      if (!result.data) {
        return {
          organic: [],
          topStories: [],
          images: [],
          relatedSearches: [],
        };
      }

      if (
        result.data.topStories != null &&
        result.data.topStories.length > numElements
      ) {
        /** Merged news results can far exceed the requested source count;
         * every entry is formatted into the LLM output, so cap them up
         * front — before any early return below and before scraping
         * entries the cap would discard */
        result.data.topStories = result.data.topStories.slice(0, numElements);
      }

      if (!result.data.organic) {
        return result.data;
      }

      if (!proMode) {
        const wikiSources = result.data.organic.filter((source) =>
          source.link.includes('wikipedia.org')
        );

        if (!wikiSources.length) {
          return result.data;
        }

        const wikiSourceMap = new Map<string, t.ValidSource>();
        wikiSourceMap.set(wikiSources[0].link, wikiSources[0]);
        const onContentScraped = createSourceUpdateCallback(wikiSourceMap);
        await fetchContents({
          query,
          metrics,
          target: 1,
          onGetHighlights,
          onContentScraped,
          links: [wikiSources[0].link],
        });

        for (let i = 0; i < result.data.organic.length; i++) {
          const source = result.data.organic[i];
          const updatedSource = wikiSourceMap.get(source.link);
          if (updatedSource) {
            result.data.organic[i] = {
              ...source,
              ...updatedSource,
            };
          }
        }

        return result.data;
      }

      const sourceMap = new Map<string, t.ValidSource>();
      const organicLinksSet = new Set<string>();

      // Collect organic links
      const organicLinks = collectLinks(
        result.data.organic,
        sourceMap,
        organicLinksSet
      );

      // Collect top story links, excluding any that are already in organic links
      const topStories = result.data.topStories ?? [];
      const topStoryLinks = collectLinks(
        topStories,
        sourceMap,
        organicLinksSet
      );

      if (organicLinks.length === 0 && (topStoryLinks.length === 0 || !news)) {
        return result.data;
      }

      const onContentScraped = createSourceUpdateCallback(sourceMap);
      const promises: Promise<void>[] = [];

      // Process organic links
      if (organicLinks.length > 0) {
        promises.push(
          fetchContents({
            query,
            metrics,
            onGetHighlights,
            onContentScraped,
            links: organicLinks,
            target: numElements,
          })
        );
      }

      // Process top story links
      if (news && topStoryLinks.length > 0) {
        promises.push(
          fetchContents({
            query,
            metrics,
            onGetHighlights,
            onContentScraped,
            links: topStoryLinks,
            target: numElements,
          })
        );
      }

      await Promise.all(promises);

      if (result.data.organic.length > 0) {
        updateSourcesWithContent(result.data.organic, sourceMap);
      }

      if (news && topStories.length > 0) {
        updateSourcesWithContent(topStories, sourceMap);
      }

      return result.data;
    } catch (error) {
      logger_.error('Error in processSources:', error);
      return {
        organic: [],
        topStories: [],
        images: [],
        relatedSearches: [],
        ...result.data,
        error: error instanceof Error ? error.message : String(error),
      };
    } finally {
      if (ownerMetrics == null) {
        metrics.flush();
      }
    }
  };

  return {
    processSources,
    topResults,
  };
};

/** Helper function to collect links and update sourceMap */
function collectLinks(
  sources: Array<t.OrganicResult | t.TopStoryResult>,
  sourceMap: Map<string, t.ValidSource>,
  existingLinksSet?: Set<string>
): string[] {
  const links: string[] = [];

  for (const source of sources) {
    if (source.link) {
      // For topStories, only add if not already in organic links
      if (existingLinksSet && existingLinksSet.has(source.link)) {
        continue;
      }

      links.push(source.link);
      if (existingLinksSet) {
        existingLinksSet.add(source.link);
      }
      sourceMap.set(source.link, source as t.ValidSource);
    }
  }

  return links;
}

/** Helper function to update sources with scraped content */
function updateSourcesWithContent<T extends t.ValidSource>(
  sources: T[],
  sourceMap: Map<string, t.ValidSource>
): void {
  for (let i = 0; i < sources.length; i++) {
    const source = sources[i];
    const updatedSource = sourceMap.get(source.link);
    if (updatedSource) {
      sources[i] = {
        ...source,
        ...updatedSource,
      } as T;
    }
  }
}

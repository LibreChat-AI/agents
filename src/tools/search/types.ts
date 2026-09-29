import type { RunnableConfig } from '@langchain/core/runnables';
import type { Logger as WinstonLogger } from 'winston';
import type { Agent as HttpsAgent } from 'https';
import type { Agent as HttpAgent } from 'http';
import type { BaseReranker } from './rerankers';
import { DATE_RANGE } from './schema';

export type { HttpAgent, HttpsAgent };

export interface HttpAgentConfig {
  httpAgent?: HttpAgent;
  httpsAgent?: HttpsAgent;
}

/** Transport and credential fields shared by every provider scraper config. */
export interface BaseSearchProviderConfig extends HttpAgentConfig {
  apiKey?: string;
  apiUrl?: string;
  timeout?: number;
  logger?: Logger;
}

export type SearchProvider =
  | 'serper'
  | 'searxng'
  | 'tavily'
  | 'keenable'
  | 'crw';
export type ScraperProvider =
  | 'firecrawl'
  | 'serper'
  | 'tavily'
  | 'crw'
  | 'keenable';
export type RerankerType =
  | 'infinity'
  | 'jina'
  | 'cohere'
  | 'rag-api'
  | 'custom'
  | 'none';

export interface Highlight {
  score: number;
  text: string;
  references?: UsedReferences;
}

export type ProcessedSource = {
  content?: string;
  attribution?: string;
  references?: References;
  highlights?: Highlight[];
  processed?: boolean;
};

export type ProcessedOrganic = OrganicResult & ProcessedSource;
export type ProcessedTopStory = TopStoryResult & ProcessedSource;
export type ValidSource = ProcessedOrganic | ProcessedTopStory;

export type ResultReference = {
  link: string;
  type: 'link' | 'image' | 'video';
  title?: string;
  attribution?: string;
};
export interface SearchResultData {
  turn?: number;
  organic?: ProcessedOrganic[];
  topStories?: ProcessedTopStory[];
  images?: ImageResult[];
  videos?: VideoResult[];
  places?: PlaceResult[];
  news?: NewsResult[];
  shopping?: ShoppingResult[];
  knowledgeGraph?: KnowledgeGraphResult;
  answerBox?: AnswerBoxResult;
  peopleAlsoAsk?: PeopleAlsoAskResult[];
  relatedSearches?: Array<{ query: string }>;
  references?: ResultReference[];
  error?: string;
}

export interface SearchResult {
  data?: SearchResultData;
  error?: string;
  success: boolean;
}

export interface Source {
  link: string;
  html?: string;
  title?: string;
  snippet?: string;
  date?: string;
}

export type TavilyTimeRange = 'day' | 'week' | 'month' | 'year';
export type TavilyTimeRangeInput =
  | TavilyTimeRange
  | 'h'
  | 'd'
  | 'w'
  | 'm'
  | 'y';

export interface TavilySearchOptions extends HttpAgentConfig {
  searchDepth?: 'basic' | 'advanced' | 'fast' | 'ultra-fast';
  maxResults?: number;
  includeImages?: boolean;
  includeAnswer?: boolean | 'basic' | 'advanced';
  includeRawContent?: boolean | 'markdown' | 'text';
  includeDomains?: string[];
  excludeDomains?: string[];
  topic?: 'general' | 'news' | 'finance';
  timeRange?: TavilyTimeRangeInput;
  includeImageDescriptions?: boolean;
  includeFavicon?: boolean;
  chunksPerSource?: number;
  safeSearch?: boolean;
  timeout?: number;
}

export interface TavilySearchPayload {
  query: string;
  search_depth: NonNullable<TavilySearchOptions['searchDepth']>;
  topic: NonNullable<TavilySearchOptions['topic']>;
  max_results: number;
  safe_search?: boolean;
  time_range?: TavilyTimeRange;
  country?: string;
  include_images?: boolean;
  include_answer?: NonNullable<TavilySearchOptions['includeAnswer']>;
  include_raw_content?: NonNullable<TavilySearchOptions['includeRawContent']>;
  include_domains?: string[];
  exclude_domains?: string[];
  include_image_descriptions?: boolean;
  include_favicon?: boolean;
  chunks_per_source?: number;
}

export interface CrwSearchOptions extends HttpAgentConfig {
  /** Max results to request (maps to `limit`; clamped 1..20). */
  maxResults?: number;
  /** Add 'images' to the sources array. */
  includeImages?: boolean;
  timeout?: number;
}

export type CrwSearchSource = 'web' | 'images' | 'news';

export interface CrwSearchPayload {
  query: string;
  limit: number;
  sources?: CrwSearchSource[];
  tbs?: string;
}

export interface CrwSearchResult {
  title?: string;
  url?: string;
  description?: string;
  snippet?: string;
  position?: number;
  category?: string;
}

export interface CrwImageSearchResult extends CrwSearchResult {
  imageUrl?: string;
  imageFormat?: string;
}

export interface CrwNewsSearchResult extends CrwSearchResult {
  publishedDate?: string;
}

export interface CrwSearchGroups {
  web?: CrwSearchResult[];
  images?: CrwImageSearchResult[];
  news?: CrwNewsSearchResult[];
}

export interface CrwSearchResponse {
  success: boolean;
  data?: (CrwSearchGroups & { results?: CrwSearchGroups }) | CrwSearchResult[];
  error?: string;
  error_code?: string;
}

export interface SearxNGSearchOptions extends HttpAgentConfig {
  /**
   * Comma-separated list of engines to query, e.g. "google,bing,startpage".
   * Defaults to "google,bing,duckduckgo". Engines not enabled on the target
   * instance are ignored by SearXNG.
   */
  engines?: string;
  /** Code of the language for search results. Defaults to "all". */
  language?: string;
  /** Time range filter, sent to SearXNG as `time_range`. Omitted when unset. */
  timeRange?: SearxNGSearchPayload['time_range'];
  /** HTTP request timeout in ms. Defaults to 10000. */
  timeout?: number;
}

export interface SearchConfig extends HttpAgentConfig {
  searchProvider?: SearchProvider;
  serperApiKey?: string;
  searxngInstanceUrl?: string;
  searxngApiKey?: string;
  searxngSearchOptions?: SearxNGSearchOptions;
  tavilyApiKey?: string;
  tavilySearchUrl?: string;
  tavilyExtractUrl?: string;
  tavilySearchOptions?: TavilySearchOptions;
  keenableApiKey?: string;
  keenableApiUrl?: string;
  keenableSearchOptions?: KeenableSearchOptions;
  crwApiKey?: string;
  crwApiUrl?: string;
  crwSearchOptions?: CrwSearchOptions;
}

export interface KeenableSearchOptions extends HttpAgentConfig {
  maxResults?: number;
  /** Restrict results to a single domain, e.g. "github.com". */
  site?: string;
  /** Sent as the X-Keenable-Title attribution header. Defaults to "LibreChat". */
  attributionTitle?: string;
  timeout?: number;
}

export interface KeenableSearchPayload {
  query: string;
  site?: string;
  published_after?: string;
}

export interface KeenableSearchResult {
  title?: string;
  url?: string;
  description?: string;
  snippet?: string;
  published_at?: string;
}

export interface KeenableSearchResponse {
  results?: KeenableSearchResult[];
}

export interface KeenableScraperConfig extends BaseSearchProviderConfig {
  /** Override the fetch endpoint base (default: public keyless, keyed when a
   * key is set). Env fallback: KEENABLE_FETCH_URL. */
  apiUrl?: string;
  /** Sent as the X-Keenable-Title attribution header. Defaults to "LibreChat". */
  attributionTitle?: string;
}

export type KeenableScrapeOptions = Omit<
  KeenableScraperConfig,
  'apiKey' | 'apiUrl' | 'logger' | 'httpAgent' | 'httpsAgent'
>;

/** Raw JSON shape returned by GET /v1/fetch{,/public}?url=... */
export interface KeenableFetchResult {
  url?: string;
  title?: string;
  content?: string;
  description?: string;
}

export interface KeenableScrapeResponse {
  success: boolean;
  data?: {
    content: string;
    title?: string;
    description?: string;
    url?: string;
  };
  error?: string;
}

export type References = {
  links: MediaReference[];
  images: MediaReference[];
  videos: MediaReference[];
};
export interface ScrapeResult {
  url: string;
  error?: boolean;
  content: string;
  attribution?: string;
  references?: References;
  highlights?: Highlight[];
}

export interface ProcessSourcesConfig {
  topResults?: number;
  /** Max chars of scraped content stored per source and passed to the
   * chunker/reranker. Defaults to 50,000; also configurable via the
   * `SEARCH_MAX_CONTENT_LENGTH` env var. */
  maxContentLength?: number;
  /** Chunk size (chars) for splitting scraped content before reranking.
   * Defaults to 150; also configurable via the `SEARCH_CHUNK_SIZE` env var.
   * Larger chunks send fewer documents to the reranker (lower cost/latency);
   * highlights are expanded ±300-450 chars around each hit either way. */
  chunkSize?: number;
  /** Overlap (chars) between adjacent chunks. Defaults to 50; also
   * configurable via the `SEARCH_CHUNK_OVERLAP` env var. Clamped below
   * `chunkSize`. */
  chunkOverlap?: number;
  mainExpandBy?: number;
  separatorExpandBy?: number;
  strategies?: string[];
  filterContent?: boolean;
  reranker?: BaseReranker;
  logger?: Logger;
}

export interface FirecrawlConfig {
  firecrawlApiKey?: string;
  firecrawlApiUrl?: string;
  firecrawlVersion?: string;
  firecrawlOptions?: FirecrawlScraperConfig;
}

export interface SerperScraperConfig extends BaseSearchProviderConfig {
  includeMarkdown?: boolean;
}

export interface TavilyScraperConfig extends BaseSearchProviderConfig {
  extractDepth?: 'basic' | 'advanced';
  includeImages?: boolean;
  includeFavicon?: boolean;
  format?: 'markdown' | 'text';
}

export interface ScraperContentResult {
  content: string;
}

export interface ScraperExtractionResult {
  no_extraction: ScraperContentResult;
}

export interface JinaRerankerResult {
  index: number;
  relevance_score: number;
  document?: string | { text: string };
}

export interface JinaRerankerResponse {
  model: string;
  /** Telemetry only, and absent from some Jina-compatible endpoints — never
   * dereference it on a path that would discard a usable ranking. */
  usage?: {
    total_tokens: number;
  };
  results: JinaRerankerResult[];
}

export interface CohereRerankerResult {
  index: number;
  relevance_score: number;
}

export interface CohereRerankerResponse {
  results: CohereRerankerResult[];
  id: string;
  /** Telemetry only; see {@link JinaRerankerResponse.usage}. */
  meta?: {
    api_version?: {
      version: string;
      is_experimental: boolean;
    };
    billed_units?: {
      search_units: number;
    };
  };
}

/** Mints a short-lived JWT per call; rag_api tokens are never cached or
 * reused across requests by the reranker itself. Receives the rerank
 * deadline's `AbortSignal` so a supplier minting its token over the network
 * can cancel that request when the deadline fires instead of leaving it
 * running past its caller. The argument is optional: zero-argument suppliers
 * remain valid. */
export type RagApiTokenSupplier = (
  signal?: AbortSignal
) => string | Promise<string>;

export interface RagApiRerankCandidate {
  id: string;
  text: string;
  base_score: number;
}

export interface RagApiRerankRequestBody {
  profile: string;
  query: string;
  candidates: RagApiRerankCandidate[];
  top_n: number;
}

export interface RagApiRerankResult {
  id: string;
  index: number;
  score: number;
}

export interface RagApiRerankResponse {
  profile: string;
  model: string;
  results: RagApiRerankResult[];
}

export type SafeSearchLevel = 0 | 1 | 2;

export type Logger = WinstonLogger;

/** Compact, redacted view of a thrown error, safe to hand to a logger. */
export interface SafeErrorLog {
  message: string;
  name?: string;
  code?: string;
  status?: number;
  method?: string;
  url?: string;
  responseDataSummary?: string;
  value?: string;
}

/** Why a rerank returned the candidates' original order instead of a ranking. */
export type RerankFallback =
  | 'no_api_key'
  | 'no_base_url'
  | 'no_token_supplier'
  | 'bad_response'
  | 'invalid_results'
  | 'chunk_error'
  | 'placeholder'
  | 'error';

/** One provider query. `results` is the row count that query contributed. */
export interface SearchObservation {
  provider: string;
  type: string;
  results: number;
  durationMs: number;
  error?: string;
  /** The query rejected rather than reporting failure in its response. Those
   * were logged at error level before they were aggregated, so the summary
   * has to carry the distinction to keep that severity. */
  thrown?: boolean;
}

/** One scraped link. A failure carries `error`; a success carries the sizes
 * the scrape produced, so the summary can report both without a second pass. */
export interface ScrapeObservation {
  url: string;
  chars?: number;
  highlights?: number;
  error?: string;
}

/** One reranker round trip. A search reranks once per scraped source, so
 * these fold into a single summary rather than logging per source. */
export interface RerankObservation {
  provider: string;
  chunks: number;
  results: number;
  durationMs: number;
  model?: string;
  /** Provider-reported usage: Jina tokens, Cohere billed search units. */
  units?: number;
  /** Chunks a provider candidate cap dropped before submission. */
  dropped?: number;
  /** Set only when a provider cap reduced the requested result count, so a
   * search returning fewer highlights than configured says why. */
  topK?: number;
  topKLimit?: number;
  reason?: RerankFallback;
  error?: SafeErrorLog;
}

/** Per-rerank state threaded from the start of a call to whichever exit it
 * takes, so every path records exactly one observation. */
export interface RerankRun {
  metrics: SearchMetrics;
  documents: string[];
  topK: number;
  startedAt: number;
  model?: string;
  units?: number;
  dropped?: number;
  topKLimit?: number;
}

/**
 * Fold-as-you-go counters for one `web_search` call. Recording is O(1) and
 * allocation-free past a bounded reason map; {@link SearchMetrics.flush}
 * emits at most one line per phase that actually ran.
 */
export interface SearchMetrics {
  recordSearch(observation: SearchObservation): void;
  recordScrape(observation: ScrapeObservation): void;
  recordRerank(observation: RerankObservation): void;
  flush(): void;
}
export interface SearchToolConfig
  extends SearchConfig,
    ProcessSourcesConfig,
    FirecrawlConfig {
  tavilyScraperOptions?: TavilyScraperConfig;
  crwScraperOptions?: CrwScraperConfig;
  keenableScraperOptions?: KeenableScraperConfig;
  /** Max chars of highlight content this tool feeds the MODEL per search (the
   * dominant, otherwise-unbounded part of the output). Distinct from
   * `maxContentLength`, which caps scraped/reranked content per source — full
   * content always remains in the `WEB_SEARCH` artifact. Defaults to 50,000;
   * also configurable via the `SEARCH_MAX_LLM_OUTPUT_CHARS` env var. Hosts that
   * know the context window (e.g. LibreChat) pass a window-relative value. */
  maxOutputChars?: number;
  logger?: Logger;
  safeSearch?: SafeSearchLevel;
  jinaApiKey?: string;
  jinaApiUrl?: string;
  cohereApiKey?: string;
  cohereApiUrl?: string;
  /** Base URL of the public `rag_api` service (`RAG_API_URL` env fallback).
   * Requests post to `${ragApiUrl}/v1/rerank`. */
  ragApiUrl?: string;
  /** Resolves a short-lived rag_api JWT immediately before each rerank call.
   * Required for the `'rag-api'` reranker type — without it, reranking falls
   * back to default (unranked) ordering. */
  ragApiTokenSupplier?: RagApiTokenSupplier;
  /** rag_api rerank profile. Defaults to `'fast-v1'`. */
  ragApiProfile?: string;
  customRerankerApiUrl?: string;
  customRerankerApiKey?: string;
  customRerankerModel?: string;
  rerankerType?: RerankerType;
  /** Timeout (ms) for rerank API requests. Defaults to 10,000. */
  rerankerTimeout?: number;
  scraperProvider?: ScraperProvider;
  scraperTimeout?: number;
  serperScraperOptions?: SerperScraperConfig;
  onSearchResults?: (
    results: SearchResult,
    runnableConfig?: RunnableConfig
  ) => void;
  onGetHighlights?: (link: string) => void;
}
export interface MediaReference {
  originalUrl: string;
  title?: string;
  text?: string;
}

export type UsedReferences = {
  type: 'link' | 'image' | 'video';
  originalIndex: number;
  reference: MediaReference;
}[];

export type AnyScraperResponse =
  | FirecrawlScrapeResponse
  | SerperScrapeResponse
  | TavilyScrapeResponse
  | CrwScrapeResponse
  | KeenableScrapeResponse;

/** Base Scraper Interface */
export interface BaseScraper {
  scrapeUrl(
    url: string,
    options?: unknown
  ): Promise<[string, AnyScraperResponse]>;
  scrapeUrls?(
    urls: string[],
    options?: unknown
  ): Promise<Array<[string, AnyScraperResponse]>>;
  extractContent(
    response: AnyScraperResponse
  ): [string, undefined | References];
  extractMetadata(
    response: AnyScraperResponse
  ): ScrapeMetadata | GenericScrapeMetadata;
}

/** Firecrawl */
export type FirecrawlScrapeOptions = Omit<
  FirecrawlScraperConfig,
  'apiKey' | 'apiUrl' | 'version' | 'logger' | 'httpAgent' | 'httpsAgent'
>;

export interface CrwScraperConfig extends BaseSearchProviderConfig {
  formats?: string[];
  onlyMainContent?: boolean;
  includeTags?: string[];
  excludeTags?: string[];
  waitFor?: number;
  headers?: Record<string, string>;
  renderJs?: boolean | null;
  cssSelector?: string;
  xpath?: string;
  proxy?: string;
  stealth?: boolean;
}

export type CrwScrapeOptions = Omit<
  CrwScraperConfig,
  'apiKey' | 'apiUrl' | 'logger' | 'httpAgent' | 'httpsAgent'
>;

export type SerperScrapeOptions = Omit<
  SerperScraperConfig,
  'apiKey' | 'apiUrl' | 'logger' | 'httpAgent' | 'httpsAgent'
>;

export type TavilyScrapeOptions = Omit<
  TavilyScraperConfig,
  'apiKey' | 'apiUrl' | 'logger' | 'httpAgent' | 'httpsAgent'
>;

export interface TavilyExtractPayload {
  urls: string[];
  extract_depth: NonNullable<TavilyScraperConfig['extractDepth']>;
  include_images: boolean;
  include_favicon?: boolean;
  format?: NonNullable<TavilyScraperConfig['format']>;
  timeout?: number;
}

export type GenericScrapeMetadata = Record<
  string,
  string | number | boolean | null | undefined
>;

export interface ScrapeMetadata {
  // Core source information
  sourceURL?: string;
  url?: string;
  scrapeId?: string;
  statusCode?: number;
  // Basic metadata
  title?: string;
  description?: string;
  language?: string;
  favicon?: string;
  viewport?: string;
  robots?: string;
  'theme-color'?: string;
  // Open Graph metadata
  'og:url'?: string;
  'og:title'?: string;
  'og:description'?: string;
  'og:type'?: string;
  'og:image'?: string;
  'og:image:width'?: string;
  'og:image:height'?: string;
  'og:site_name'?: string;
  ogUrl?: string;
  ogTitle?: string;
  ogDescription?: string;
  ogImage?: string;
  ogSiteName?: string;
  // Article metadata
  'article:author'?: string;
  'article:published_time'?: string;
  'article:modified_time'?: string;
  'article:section'?: string;
  'article:tag'?: string;
  'article:publisher'?: string;
  publishedTime?: string;
  modifiedTime?: string;
  // Twitter metadata
  'twitter:site'?: string | boolean | number | null;
  'twitter:creator'?: string;
  'twitter:card'?: string;
  'twitter:image'?: string;
  'twitter:dnt'?: string;
  'twitter:app:name:iphone'?: string;
  'twitter:app:id:iphone'?: string;
  'twitter:app:url:iphone'?: string;
  'twitter:app:name:ipad'?: string;
  'twitter:app:id:ipad'?: string;
  'twitter:app:url:ipad'?: string;
  'twitter:app:name:googleplay'?: string;
  'twitter:app:id:googleplay'?: string;
  'twitter:app:url:googleplay'?: string;
  // Facebook metadata
  'fb:app_id'?: string;
  // App links
  'al:ios:url'?: string;
  'al:ios:app_name'?: string;
  'al:ios:app_store_id'?: string;
  // Allow for additional properties that might be present
  [key: string]: string | number | boolean | null | undefined;
}

export interface FirecrawlScrapeResponse {
  success: boolean;
  data?: {
    markdown?: string;
    html?: string;
    rawHtml?: string;
    screenshot?: string;
    links?: string[];
    metadata?: ScrapeMetadata;
  };
  error?: string;
}

export interface CrwScrapeData {
  markdown?: string;
  html?: string;
  rawHtml?: string;
  plainText?: string;
  screenshot?: string;
  links?: string[];
  metadata?: ScrapeMetadata;
}

export interface CrwScrapeResponse {
  success: boolean;
  data?: CrwScrapeData;
  error?: string;
  error_code?: string;
}

export interface CrwRawScrapeResponse extends CrwScrapeData {
  success?: boolean;
  data?: CrwScrapeData;
  error?: string;
  error_code?: string;
}

export interface SerperScrapeResponse {
  success: boolean;
  data?: {
    text?: string;
    markdown?: string;
    metadata?: Record<string, string | number | boolean | null | undefined>;
    credits?: number;
  };
  error?: string;
}

export interface TavilyScrapeResponse {
  success: boolean;
  data?: {
    rawContent?: string;
    images?: string[];
    favicon?: string;
  };
  error?: string;
}

export interface TavilySearchResult {
  title?: string;
  url?: string;
  content?: string;
  score?: number;
  published_date?: string;
}

export type TavilyImageResult =
  | string
  | {
      url?: string;
      description?: string;
    };

export interface TavilySearchResponse {
  answer?: string;
  images?: TavilyImageResult[];
  results?: TavilySearchResult[];
}

export interface TavilyExtractResult {
  url: string;
  raw_content?: string;
  images?: string[];
  favicon?: string;
  error?: string;
}

export interface FirecrawlScraperConfig extends BaseSearchProviderConfig {
  version?: string;
  formats?: string[];
  includeTags?: string[];
  excludeTags?: string[];
  waitFor?: number;
  maxAge?: number;
  mobile?: boolean;
  skipTlsVerification?: boolean;
  blockAds?: boolean;
  removeBase64Images?: boolean;
  parsePDF?: boolean;
  storeInCache?: boolean;
  zeroDataRetention?: boolean;
  headers?: Record<string, string>;
  location?: { country?: string; languages?: string[] };
  onlyMainContent?: boolean;
  changeTrackingOptions?: object;
}

/** Result kind a parallel sub-search covers; `web` is the untyped main
 * search, which every provider serves without a `type` parameter. */
export type SubSearchType = 'web' | 'images' | 'videos' | 'news';

export type GetSourcesParams = {
  query: string;
  date?: DATE_RANGE;
  country?: string;
  numResults?: number;
  safeSearch?: SearchToolConfig['safeSearch'];
  images?: boolean;
  videos?: boolean;
  news?: boolean;
  type?: 'search' | 'images' | 'videos' | 'news';
};

/** Serper API */
export interface VideoResult {
  title?: string;
  link?: string;
  snippet?: string;
  imageUrl?: string;
  duration?: string;
  source?: string;
  channel?: string;
  date?: string;
  position?: number;
}

export interface PlaceResult {
  position?: number;
  name?: string;
  address?: string;
  latitude?: number;
  longitude?: number;
  rating?: number;
  ratingCount?: number;
  category?: string;
  identifier?: string;
}

export interface NewsResult {
  title?: string;
  link?: string;
  snippet?: string;
  date?: string;
  source?: string;
  imageUrl?: string;
  position?: number;
}

export interface ShoppingResult {
  title?: string;
  source?: string;
  link?: string;
  price?: string;
  delivery?: string;
  imageUrl?: string;
  rating?: number;
  ratingCount?: number;
  offers?: string;
  productId?: string;
  position?: number;
}

export interface ScholarResult {
  title?: string;
  link?: string;
  publicationInfo?: string;
  snippet?: string;
  year?: number;
  citedBy?: number;
}

export interface ImageResult {
  title?: string;
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
  thumbnailUrl?: string;
  thumbnailWidth?: number;
  thumbnailHeight?: number;
  source?: string;
  domain?: string;
  link?: string;
  googleUrl?: string;
  position?: number;
}

export interface SerperSearchPayload extends SerperSearchInput {
  /**
   * Search type/vertical
   * Options: "search" (web), "images", "news", "places", "videos"
   */
  type?: 'search' | 'images' | 'news' | 'places' | 'videos';

  /**
   * Starting index for search results pagination (used instead of page)
   */
  start?: number;

  /**
   * Filtering for safe search
   * Options: "off", "moderate", "active"
   */
  safe?: 'off' | 'moderate' | 'active';
}

export type SerperSearchParameters = Pick<SerperSearchPayload, 'q' | 'type'> & {
  engine: 'google';
};

export interface OrganicResult {
  position?: number;
  title?: string;
  link: string;
  snippet?: string;
  date?: string;
  sitelinks?: Array<{
    title: string;
    link: string;
  }>;
}

export interface TopStoryResult {
  title?: string;
  link: string;
  source?: string;
  date?: string;
  imageUrl?: string;
}
export interface KnowledgeGraphResult {
  title?: string;
  type?: string;
  imageUrl?: string;
  description?: string;
  descriptionSource?: string;
  descriptionLink?: string;
  attributes?: Record<string, string>;
  website?: string;
}

export interface AnswerBoxResult {
  title?: string;
  snippet?: string;
  snippetHighlighted?: string[];
  link?: string;
  date?: string;
}

export interface PeopleAlsoAskResult {
  question?: string;
  snippet?: string;
  title?: string;
  link?: string;
}

export type RelatedSearches = Array<{ query: string }>;

export interface SerperSearchInput {
  /**
   * The search query string
   */
  q: string;

  /**
   * Country code for localized results
   * Examples: "us", "uk", "ca", "de", etc.
   */
  gl?: string;

  /**
   * Interface language
   * Examples: "en", "fr", "de", etc.
   */
  hl?: string;

  /**
   * Number of results to return (up to 100)
   */
  num?: number;
  /**
   * Specific location for contextual results
   * Example: "New York, NY"
   */
  location?: string;

  /**
   * Search autocorrection setting
   */
  autocorrect?: boolean;
  page?: number;
  /**
   * Date range for search results
   * Options: "h" (past hour), "d" (past 24 hours), "w" (past week),
   * "m" (past month), "y" (past year)
   * `qdr:${DATE_RANGE}`
   */
  tbs?: string;
}

export type SerperResultData = {
  searchParameters: SerperSearchPayload;
  organic?: OrganicResult[];
  topStories?: TopStoryResult[];
  images?: ImageResult[];
  videos?: VideoResult[];
  places?: PlaceResult[];
  news?: NewsResult[];
  shopping?: ShoppingResult[];
  peopleAlsoAsk?: PeopleAlsoAskResult[];
  relatedSearches?: RelatedSearches;
  knowledgeGraph?: KnowledgeGraphResult;
  answerBox?: AnswerBoxResult;
  credits?: number;
};

/** SearXNG */

export interface SearxNGSearchPayload {
  /**
   * The search query string
   * Supports syntax specific to different search engines
   * Example: "site:github.com SearXNG"
   */
  q: string;

  /**
   * Comma-separated list of search categories
   * Example: "general,images,news"
   */
  categories?: string;

  /**
   * Comma-separated list of search engines to use
   * Example: "google,bing,duckduckgo"
   */
  engines?: string;

  /**
   * Code of the language for search results
   * Example: "en", "fr", "de", "es"
   */
  language?: string;

  /**
   * Search page number
   * Default: 1
   */
  pageno?: number;

  /**
   * Time range filter for search results
   * Options: "day", "month", "year"
   */
  time_range?: 'day' | 'month' | 'year';

  /**
   * Output format of results
   * Options: "json", "csv", "rss"
   */
  format?: 'json' | 'csv' | 'rss';

  /**
   * Open search results on new tab
   * Options: `0` (off), `1` (on)
   */
  results_on_new_tab?: 0 | 1;

  /**
   * Proxy image results through SearxNG
   * Options: true, false
   */
  image_proxy?: boolean;

  /**
   * Service for autocomplete suggestions
   * Options: "google", "dbpedia", "duckduckgo", "mwmbl",
   *          "startpage", "wikipedia", "stract", "swisscows", "qwant"
   */
  autocomplete?: string;

  /**
   * Safe search filtering level
   * Options: "0" (off), "1" (moderate), "2" (strict)
   */
  safesearch?: 0 | 1 | 2;

  /**
   * Theme to use for results page
   * Default: "simple" (other themes may be available per instance)
   */
  theme?: string;

  /**
   * List of enabled plugins
   * Default: "Hash_plugin,Self_Information,Tracker_URL_remover,Ahmia_blacklist"
   */
  enabled_plugins?: string;

  /**
   * List of disabled plugins
   */
  disabled_plugins?: string;

  /**
   * List of enabled engines
   */
  enabled_engines?: string;

  /**
   * List of disabled engines
   */
  disabled_engines?: string;
}

export interface SearXNGResult {
  title?: string;
  url?: string;
  content?: string;
  publishedDate?: string;
  img_src?: string;
  score?: number;
  engine?: string;
  category?: string;
  thumbnail?: string;
  priority?: string;
  engines?: string[];
  positions?: number[];
  template?: string;
  parsed_url?: string[];
}

export type ProcessSourcesFields = {
  result: SearchResult;
  numElements: number;
  query: string;
  news: boolean;
  proMode: boolean;
  onGetHighlights: SearchToolConfig['onGetHighlights'];
  /** Collector owned by the caller; when omitted, one is created and flushed
   * for this call so a direct `processSources` still summarizes itself. */
  metrics?: SearchMetrics;
};

export interface SearchToolSchema {
  type: 'object';
  properties: Record<string, unknown>;
  required: string[];
}

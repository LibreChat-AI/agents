import axios from 'axios';
import type { AxiosResponse } from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_KEENABLE_SCRAPE_TIMEOUT = 15000;

/** Keyed and keyless fetch endpoints. Keenable reads any URL as clean markdown
 * without a key via the public endpoint; a key only lifts rate limits. */
const KEENABLE_FETCH_API_URL = 'https://api.keenable.ai/v1/fetch';
const KEENABLE_FETCH_PUBLIC_URL = 'https://api.keenable.ai/v1/fetch/public';

/** Keenable enforces a per-second rate limit and answers bursts with 429 and
 * `Retry-After`; such requests are retried a bounded number of times. */
const KEENABLE_MAX_RATE_LIMIT_RETRIES = 2;
const KEENABLE_DEFAULT_RETRY_DELAY_MS = 1000;
const KEENABLE_MAX_RETRY_DELAY_MS = 3000;
const KEENABLE_RETRY_JITTER_MS = 500;

/** Delay before retrying a 429 response, or undefined for any other failure. */
const getRateLimitRetryDelayMs = (error: unknown): number | undefined => {
  if (!axios.isAxiosError(error) || error.response?.status !== 429) {
    return undefined;
  }
  const retryAfterSeconds = Number(error.response.headers['retry-after']);
  const baseDelayMs =
    Number.isFinite(retryAfterSeconds) && retryAfterSeconds >= 0
      ? Math.min(retryAfterSeconds * 1000, KEENABLE_MAX_RETRY_DELAY_MS)
      : KEENABLE_DEFAULT_RETRY_DELAY_MS;
  return baseDelayMs + Math.random() * KEENABLE_RETRY_JITTER_MS;
};

export class KeenableScraper implements t.BaseScraper {
  private apiKey: string | undefined;
  private apiUrl: string;
  private timeout: number;
  private attributionTitle: string;
  private logger: t.Logger;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor(config: t.KeenableScraperConfig = {}) {
    const resolvedKey = config.apiKey ?? process.env.KEENABLE_API_KEY;
    this.apiKey =
      resolvedKey != null && resolvedKey !== '' ? resolvedKey : undefined;
    this.apiUrl =
      config.apiUrl ??
      process.env.KEENABLE_FETCH_URL ??
      (this.apiKey != null
        ? KEENABLE_FETCH_API_URL
        : KEENABLE_FETCH_PUBLIC_URL);
    this.timeout = config.timeout ?? DEFAULT_KEENABLE_SCRAPE_TIMEOUT;
    this.attributionTitle = config.attributionTitle ?? 'LibreChat';
    this.httpAgent = config.httpAgent;
    this.httpsAgent = config.httpsAgent;
    this.logger = config.logger || createDefaultLogger();
  }

  private buildHeaders(): Record<string, string> {
    /** X-Keenable-Title is used for traffic attribution and required on the
     * keyless endpoint; the key only lifts rate limits, so it is sent only when
     * present. */
    const headers: Record<string, string> = {
      'X-Keenable-Title': this.attributionTitle,
    };
    if (this.apiKey != null) {
      headers['X-API-Key'] = this.apiKey;
    }
    return headers;
  }

  private async fetchPage(
    url: string,
    timeout: number,
    retriesLeft = KEENABLE_MAX_RATE_LIMIT_RETRIES
  ): Promise<AxiosResponse<t.KeenableFetchResult>> {
    try {
      return await axios.get<t.KeenableFetchResult>(this.apiUrl, {
        params: { url },
        headers: this.buildHeaders(),
        timeout,
        httpAgent: this.httpAgent,
        httpsAgent: this.httpsAgent,
      });
    } catch (error) {
      const delayMs = getRateLimitRetryDelayMs(error);
      if (delayMs == null || retriesLeft <= 0) {
        throw error;
      }
      this.logger.debug(
        `Keenable Fetch rate limited (429), retrying in ${Math.round(delayMs)} ms`
      );
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      return this.fetchPage(url, timeout, retriesLeft - 1);
    }
  }

  async scrapeUrl(
    url: string,
    options: t.KeenableScrapeOptions = {}
  ): Promise<[string, t.KeenableScrapeResponse]> {
    if (!url || !url.trim()) {
      return [url, { success: false, error: 'URL cannot be empty' }];
    }

    try {
      const response = await this.fetchPage(
        url,
        options.timeout ?? this.timeout
      );

      const data = response.data;
      const content = data.content ?? '';
      if (!content) {
        return [
          url,
          { success: false, error: 'Keenable Fetch returned no content' },
        ];
      }

      return [
        url,
        {
          success: true,
          data: {
            content,
            title: data.title,
            description: data.description,
            url: data.url ?? url,
          },
        },
      ];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `Keenable Fetch API request failed: ${errorMessage}`,
        },
      ];
    }
  }

  async scrapeUrls(
    urls: string[],
    options: t.KeenableScrapeOptions = {}
  ): Promise<Array<[string, t.KeenableScrapeResponse]>> {
    /** Keenable fetch is single-URL; run the batch concurrently. */
    return Promise.all(urls.map((url) => this.scrapeUrl(url, options)));
  }

  extractContent(
    response: t.KeenableScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }
    /** Keenable returns clean markdown with no separate media arrays, so there
     * are no structured references to surface. */
    return [response.data.content, undefined];
  }

  extractMetadata(response: t.KeenableScrapeResponse): t.GenericScrapeMetadata {
    if (!response.success || !response.data) {
      return {};
    }
    const metadata: t.GenericScrapeMetadata = {};
    if (response.data.title != null) {
      metadata.title = response.data.title;
    }
    if (response.data.description != null && response.data.description !== '') {
      metadata.description = response.data.description;
    }
    if (response.data.url != null) {
      metadata.url = response.data.url;
    }
    return metadata;
  }
}

export const createKeenableScraper = (
  config: t.KeenableScraperConfig = {}
): KeenableScraper => {
  return new KeenableScraper(config);
};

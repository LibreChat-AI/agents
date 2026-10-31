import axios from 'axios';
import type * as t from './types';
import { getKagiUrl, resolveKagiKey } from './kagi';

interface KagiPage {
  url: string;
  markdown?: string | null;
  error?: string;
}

interface KagiExtractResponse {
  data?: Array<KagiPage | null>;
  error?: object[];
  errors?: object[];
}

type KagiScrapeOptions = Pick<t.KagiScraperConfig, 'timeout'>;
type KagiScrapeEntry = [string, t.KagiScrapeResponse];

function failure(url: string, error: string): KagiScrapeEntry {
  return [url, { success: false, error }];
}

export class KagiScraper implements t.BaseScraper {
  private readonly apiKey: string;
  private readonly config: t.KagiScraperConfig;

  constructor(config: t.KagiScraperConfig = {}) {
    this.apiKey = resolveKagiKey(config.apiKey);
    this.config = config;
  }

  async scrapeUrl(
    url: string,
    options: KagiScrapeOptions = {}
  ): Promise<KagiScrapeEntry> {
    const [result] = await this.scrapeUrls([url], options);
    return result;
  }

  async scrapeUrls(
    urls: string[],
    options: KagiScrapeOptions = {}
  ): Promise<KagiScrapeEntry[]> {
    const batches: Promise<KagiScrapeEntry[]>[] = [];
    for (let index = 0; index < urls.length; index += 10) {
      batches.push(this.extractBatch(urls.slice(index, index + 10), options));
    }
    return (await Promise.all(batches)).flat();
  }

  private async extractBatch(
    urls: string[],
    options: KagiScrapeOptions
  ): Promise<KagiScrapeEntry[]> {
    const urlKeys = new Map<string, string>();
    const validUrls: string[] = [];
    for (const url of urls) {
      const key = getKagiUrl(url);
      if (key !== undefined) {
        urlKeys.set(url, key);
        validUrls.push(key);
      }
    }
    if (validUrls.length === 0) {
      return urls.map((url) =>
        failure(url, 'Kagi Extract requires an HTTP(S) URL without credentials')
      );
    }

    try {
      const response = await axios.post<KagiExtractResponse | null>(
        'https://kagi.com/api/v1/extract',
        { pages: validUrls.map((url) => ({ url })), format: 'json' },
        {
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
          timeout: options.timeout ?? this.config.timeout ?? 7500,
          httpAgent: this.config.httpAgent,
          httpsAgent: this.config.httpsAgent,
          maxRedirects: 0,
        }
      );
      const envelope = response.data;
      if (
        envelope?.error != null &&
        (!Array.isArray(envelope.error) || envelope.error.length > 0)
      ) {
        return urls.map((url) =>
          failure(url, 'Kagi Extract API returned an error')
        );
      }
      if (
        !Array.isArray(envelope?.data) ||
        (envelope.errors != null && !Array.isArray(envelope.errors))
      ) {
        return urls.map((url) =>
          failure(url, 'Kagi Extract API returned an invalid response')
        );
      }
      const pages = new Map<string, KagiPage>();
      for (const page of envelope.data) {
        if (page == null || typeof page.url !== 'string') {
          continue;
        }
        const key = getKagiUrl(page.url);
        if (key !== undefined) {
          pages.set(key, page);
        }
      }
      return urls.map((url) => {
        const key = urlKeys.get(url);
        if (key === undefined) {
          return failure(
            url,
            'Kagi Extract requires an HTTP(S) URL without credentials'
          );
        }
        const page = pages.get(key);
        if (
          page == null ||
          page.error != null ||
          typeof page.markdown !== 'string' ||
          page.markdown.trim() === ''
        ) {
          return failure(url, 'Kagi Extract could not extract page content');
        }
        return [url, { success: true, data: { url, markdown: page.markdown } }];
      });
    } catch {
      return urls.map((url) => failure(url, 'Kagi Extract API request failed'));
    }
  }

  extractContent(response: t.KagiScrapeResponse): [string, undefined] {
    return [response.success ? (response.data?.markdown ?? '') : '', undefined];
  }

  extractMetadata(response: t.KagiScrapeResponse): t.GenericScrapeMetadata {
    return response.success && response.data ? { url: response.data.url } : {};
  }
}

export function createKagiScraper(
  config: t.KagiScraperConfig = {}
): KagiScraper {
  return new KagiScraper(config);
}

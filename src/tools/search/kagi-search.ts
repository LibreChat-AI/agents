import axios from 'axios';
import type * as t from './types';
import { DATE_RANGE } from './schema';

interface KagiResult {
  url: string;
  title: string;
  snippet?: string;
  time?: string;
  image?: { url: string } | null;
}

interface KagiSearchResponse {
  data?: {
    search?: Array<KagiResult | null>;
    image?: Array<KagiResult | null>;
    video?: Array<KagiResult | null>;
    news?: Array<KagiResult | null>;
    related_search?: Array<KagiResult | null>;
  };
  error?: object[];
}

interface KagiSearchPayload {
  query: string;
  workflow: NonNullable<t.GetSourcesParams['type']>;
  format: 'json';
  limit: number;
  safe_search?: boolean;
  filters?: { region?: string; after?: string };
  lens?: { time_relative: 'day' | 'week' | 'month' };
}

/** Explicitly empty credentials disable environment fallback. */
export function resolveKagiKey(apiKey?: string): string {
  const key = apiKey ?? process.env.KAGI_API_KEY;
  if (key == null || key.trim() === '') {
    throw new Error('Kagi API key is required');
  }
  return key;
}

function validResults(
  results: Array<KagiResult | null>
): results is KagiResult[] {
  return results.every(
    (result) =>
      result != null &&
      typeof result.url === 'string' &&
      result.url.trim() !== '' &&
      typeof result.title === 'string' &&
      (result.snippet === undefined || typeof result.snippet === 'string') &&
      (result.time === undefined || typeof result.time === 'string') &&
      (result.image === undefined ||
        (result.image != null && typeof result.image.url === 'string'))
  );
}

export function createKagiAPI(
  apiKey?: string,
  options: t.KagiSearchOptions = {}
): { getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult> } {
  const key = resolveKagiKey(apiKey);

  const getSources = async ({
    query,
    type = 'search',
    country,
    date,
    safeSearch,
    numResults = 8,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    const requestedLimit = options.limit ?? numResults;
    const payload: KagiSearchPayload = {
      query,
      workflow: type,
      format: 'json',
      limit: Number.isFinite(requestedLimit)
        ? Math.min(1024, Math.max(1, Math.trunc(requestedLimit)))
        : 8,
    };
    if (safeSearch != null) {
      payload.safe_search = safeSearch !== 0;
    }
    if (country != null && country !== '') {
      const region = country.toUpperCase();
      payload.filters = { region: region === 'UK' ? 'GB' : region };
    }
    if (date === DATE_RANGE.PAST_YEAR) {
      const after = new Date();
      after.setUTCFullYear(after.getUTCFullYear() - 1);
      payload.filters = {
        ...payload.filters,
        after: after.toISOString().slice(0, 10),
      };
    } else if (date != null) {
      const intervals = {
        [DATE_RANGE.PAST_HOUR]: 'day',
        [DATE_RANGE.PAST_24_HOURS]: 'day',
        [DATE_RANGE.PAST_WEEK]: 'week',
        [DATE_RANGE.PAST_MONTH]: 'month',
      } as const;
      payload.lens = { time_relative: intervals[date] };
    }

    try {
      const response = await axios.post<KagiSearchResponse | null>(
        'https://kagi.com/api/v1/search',
        payload,
        {
          headers: {
            Authorization: `Bearer ${key}`,
            'Content-Type': 'application/json',
          },
          timeout: options.timeout ?? 10000,
          httpAgent: options.httpAgent,
          httpsAgent: options.httpsAgent,
          maxRedirects: 0,
        }
      );
      const envelope = response.data;
      if (
        envelope?.error != null &&
        (!Array.isArray(envelope.error) || envelope.error.length > 0)
      ) {
        return { success: false, error: 'Kagi Search API returned an error' };
      }
      const data = envelope?.data;
      const resultKeys = {
        search: 'search',
        images: 'image',
        videos: 'video',
        news: 'news',
      } as const;
      const resultKey = resultKeys[type];
      if (
        data == null ||
        Array.isArray(data) ||
        !Array.isArray(data[resultKey])
      ) {
        return {
          success: false,
          error: 'Kagi Search API returned an invalid response',
        };
      }
      const results: t.SearchResultData = { organic: [], topStories: [] };
      for (const name of [
        'search',
        'image',
        'video',
        'news',
        'related_search',
      ] as const) {
        const rows = data[name];
        if (rows === undefined) {
          continue;
        }
        if (!Array.isArray(rows) || !validResults(rows)) {
          return {
            success: false,
            error: 'Kagi Search API returned an invalid response',
          };
        }
        const limited = rows.slice(0, payload.limit);
        if (name === 'related_search') {
          results.relatedSearches = limited.map((row) => ({
            query: row.title,
          }));
        } else if (name === 'image') {
          results.images = limited.map((row, index) => ({
            title: row.title,
            link: row.url,
            imageUrl: row.image?.url ?? row.url,
            position: index + 1,
          }));
        } else {
          const mapped = limited.map((row, index) => ({
            title: row.title,
            link: row.url,
            snippet: row.snippet,
            date: row.time,
            imageUrl: row.image?.url,
            position: index + 1,
          }));
          if (name === 'search') {
            results.organic = mapped;
          } else if (name === 'video') {
            results.videos = mapped;
          } else {
            results.news = mapped;
          }
        }
      }
      return { success: true, data: results };
    } catch {
      return { success: false, error: 'Kagi Search API request failed' };
    }
  };

  return { getSources };
}

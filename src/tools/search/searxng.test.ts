import axios from 'axios';
import { Agent as HttpAgent } from 'http';
import { Agent as HttpsAgent } from 'https';
import type * as t from './types';
import { createSearchAPI } from './search';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const INSTANCE_URL = 'https://searxng.example.com';

const emptyResponse = { data: { results: [] } };

/** The params object handed to `axios.get` for the most recent call. */
const lastParams = (): t.SearxNGSearchPayload =>
  mockedAxios.get.mock.calls[0][1]?.params as t.SearxNGSearchPayload;

/** The full request config handed to `axios.get` for the most recent call. */
const lastRequestConfig = () => mockedAxios.get.mock.calls[0][1];

describe('SearXNG search options', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.SEARXNG_INSTANCE_URL;
    delete process.env.SEARXNG_API_KEY;
    mockedAxios.get.mockResolvedValue(emptyResponse);
  });

  describe('defaults when searxngSearchOptions is absent', () => {
    it('sends the original engines, language, and timeout, and no time_range', async () => {
      const searchAPI = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
      });
      await searchAPI.getSources({ query: 'typescript' });

      expect(lastParams()).toEqual({
        q: 'typescript',
        format: 'json',
        pageno: 1,
        categories: 'general',
        language: 'all',
        safesearch: undefined,
        engines: 'google,bing,duckduckgo',
      });
      expect(lastParams()).not.toHaveProperty('time_range');
      expect(lastRequestConfig()?.timeout).toBe(10000);
    });

    it('sends the same defaults when an empty options object is supplied', async () => {
      const searchAPI = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
        searxngSearchOptions: {},
      });
      await searchAPI.getSources({ query: 'typescript' });

      const params = lastParams();
      expect(params.engines).toBe('google,bing,duckduckgo');
      expect(params.language).toBe('all');
      expect(params).not.toHaveProperty('time_range');
      expect(lastRequestConfig()?.timeout).toBe(10000);
    });
  });

  describe('overrides', () => {
    it('applies engines, language, and timeout when provided', async () => {
      const searchAPI = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
        searxngSearchOptions: {
          engines: 'startpage,qwant',
          language: 'de',
          timeout: 45000,
        },
      });
      await searchAPI.getSources({ query: 'typescript' });

      const params = lastParams();
      expect(params.engines).toBe('startpage,qwant');
      expect(params.language).toBe('de');
      expect(lastRequestConfig()?.timeout).toBe(45000);
    });

    it('omits time_range when unset and sends it when set', async () => {
      const withoutRange = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
        searxngSearchOptions: { engines: 'startpage' },
      });
      await withoutRange.getSources({ query: 'typescript' });
      expect(lastParams()).not.toHaveProperty('time_range');

      jest.clearAllMocks();
      mockedAxios.get.mockResolvedValue(emptyResponse);

      const withRange = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
        searxngSearchOptions: { timeRange: 'month' },
      });
      await withRange.getSources({ query: 'typescript' });
      expect(lastParams().time_range).toBe('month');
    });

    it('falls back to the defaults for blank engines and language', async () => {
      const searchAPI = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
        searxngSearchOptions: { engines: '  ', language: '' },
      });
      await searchAPI.getSources({ query: 'typescript' });

      const params = lastParams();
      expect(params.engines).toBe('google,bing,duckduckgo');
      expect(params.language).toBe('all');
    });
  });

  describe('fields the options block must not take over', () => {
    it.each([
      ['general', undefined],
      ['images', 'images'],
      ['videos', 'videos'],
      ['news', 'news'],
    ] as const)(
      'derives categories=%s from type even with options supplied',
      async (expectedCategory, type) => {
        const searchAPI = createSearchAPI({
          searchProvider: 'searxng',
          searxngInstanceUrl: INSTANCE_URL,
          searxngSearchOptions: {
            engines: 'startpage,qwant',
            language: 'fr',
            timeRange: 'day',
          },
        });
        await searchAPI.getSources({ query: 'typescript', type });

        const params = lastParams();
        expect(params.categories).toBe(expectedCategory);
        expect(params.engines).toBe('startpage,qwant');
      }
    );

    it('sources safesearch from the top-level safeSearch argument', async () => {
      const searchAPI = createSearchAPI({
        searchProvider: 'searxng',
        searxngInstanceUrl: INSTANCE_URL,
        searxngSearchOptions: { engines: 'startpage' },
      });
      await searchAPI.getSources({ query: 'typescript', safeSearch: 2 });

      expect(lastParams().safesearch).toBe(2);
    });
  });

  it('merges configured headers into the request', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'searxng',
      searxngInstanceUrl: INSTANCE_URL,
      searxngSearchOptions: { headers: { 'x-end-user': 'user@example.com' } },
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(lastRequestConfig()?.headers).toEqual({
      'Content-Type': 'application/json',
      'x-end-user': 'user@example.com',
    });
  });

  it('still threads http agents supplied alongside the options block', async () => {
    const httpAgent = new HttpAgent();
    const httpsAgent = new HttpsAgent();

    const searchAPI = createSearchAPI({
      searchProvider: 'searxng',
      searxngInstanceUrl: INSTANCE_URL,
      searxngSearchOptions: { engines: 'startpage' },
      httpAgent,
      httpsAgent,
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(lastRequestConfig()).toEqual(
      expect.objectContaining({ httpAgent, httpsAgent })
    );
    expect(lastParams().engines).toBe('startpage');
  });
});

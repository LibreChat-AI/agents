import axios from 'axios';
import { createLogger } from 'winston';
import { Agent as HttpAgent } from 'http';
import { Agent as HttpsAgent } from 'https';
import type * as t from './types';
import { createKagiScraper } from './kagi-scraper';
import { createSearchAPI } from './search';
import { createSearchTool } from './tool';
import { DATE_RANGE } from './schema';

const searchUrl = 'https://kagi.com/api/v1/search';
const extractUrl = 'https://kagi.com/api/v1/extract';
const url = 'https://example.com/article';
const row = {
  url,
  title: 'Evidence',
  snippet: 'A useful snippet',
  time: '2026-10-01T12:00:00Z',
  image: { url: 'https://example.com/image.png' },
};
const secret = 'PRIVATE-KEY-AND-UPSTREAM-DIAGNOSTICS';
const logger = createLogger({ silent: true });
const originalKey = process.env.KAGI_API_KEY;

function searchAPI(
  options: t.KagiSearchOptions = {}
): ReturnType<typeof createSearchAPI> {
  return createSearchAPI({
    searchProvider: 'kagi',
    kagiApiKey: 'search-key',
    kagiSearchOptions: options,
  });
}

beforeEach(() => {
  delete process.env.KAGI_API_KEY;
});
afterEach(() => {
  jest.restoreAllMocks();
  if (originalKey === undefined) {
    delete process.env.KAGI_API_KEY;
  } else {
    process.env.KAGI_API_KEY = originalKey;
  }
});

describe('Kagi configuration', () => {
  it('requires a key for either provider', () => {
    expect(() => createSearchAPI({ searchProvider: 'kagi' })).toThrow(
      'Kagi API key is required'
    );
    expect(() => createKagiScraper()).toThrow('Kagi API key is required');
  });

  it('never falls back to the environment for an explicitly empty key', () => {
    process.env.KAGI_API_KEY = secret;
    expect(() =>
      createSearchAPI({ searchProvider: 'kagi', kagiApiKey: '' })
    ).toThrow('Kagi API key is required');
    expect(() => createKagiScraper({ apiKey: '' })).toThrow(
      'Kagi API key is required'
    );
    expect(() =>
      createSearchTool({
        searchProvider: 'kagi',
        scraperProvider: 'kagi',
        kagiApiKey: '',
        logger,
      })
    ).toThrow('Kagi API key is required');
    expect(() =>
      createSearchTool({
        searchProvider: 'kagi',
        scraperProvider: 'kagi',
        kagiApiKey: 'shared',
        kagiScraperOptions: { apiKey: '' },
        logger,
      })
    ).toThrow('Kagi API key is required');
  });

  it('uses the environment for omitted keys in search and extract', async () => {
    process.env.KAGI_API_KEY = 'environment-key';
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: { search: [] } } })
      .mockResolvedValueOnce({ data: { data: [{ url, markdown: '# Page' }] } });
    await createSearchAPI({ searchProvider: 'kagi' }).getSources({
      query: 'test',
    });
    await createKagiScraper().scrapeUrl(url);
    for (const call of post.mock.calls) {
      expect(call[2]?.headers?.Authorization).toBe('Bearer environment-key');
    }
  });
});

describe('Kagi Search v1', () => {
  it('posts JSON and maps web results, media, news and related searches', async () => {
    const post = jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: {
        data: {
          search: [
            row,
            { url: 'https://example.com/minimal', title: 'Minimal' },
          ],
          image: [row],
          video: [row],
          news: [row],
          related_search: [
            {
              url: 'https://kagi.com/search?q=related',
              title: 'Related query',
            },
          ],
        },
      },
    });
    const result = await searchAPI().getSources({ query: 'test' });
    expect(post).toHaveBeenCalledWith(
      searchUrl,
      {
        query: 'test',
        workflow: 'search',
        format: 'json',
        limit: 8,
      },
      expect.objectContaining({
        headers: {
          Authorization: 'Bearer search-key',
          'Content-Type': 'application/json',
        },
        timeout: 10000,
        maxRedirects: 0,
      })
    );
    expect(result.success).toBe(true);
    expect(result.data?.organic).toEqual([
      expect.objectContaining({
        link: url,
        title: row.title,
        snippet: row.snippet,
        date: row.time,
        position: 1,
      }),
      expect.objectContaining({
        title: 'Minimal',
        link: 'https://example.com/minimal',
      }),
    ]);
    expect(result.data?.images).toEqual([
      expect.objectContaining({ imageUrl: row.image.url, link: url }),
    ]);
    expect(result.data?.videos).toEqual([
      expect.objectContaining({
        link: url,
        imageUrl: row.image.url,
        date: row.time,
      }),
    ]);
    expect(result.data?.news).toEqual([
      expect.objectContaining({
        link: url,
        imageUrl: row.image.url,
        date: row.time,
      }),
    ]);
    expect(result.data?.relatedSearches).toEqual([{ query: 'Related query' }]);
  });

  it.each([
    ['images', 'image', 'images'],
    ['videos', 'video', 'videos'],
    ['news', 'news', 'news'],
  ] as const)('maps the %s workflow', async (type, bucket, output) => {
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: { [bucket]: [row] } } });
    const result = await searchAPI().getSources({ query: 'test', type });
    expect(result.success).toBe(true);
    expect(result.data?.[output]).toEqual([
      expect.objectContaining({ link: url }),
    ]);
    expect(post.mock.calls[0][1]).toEqual(
      expect.objectContaining({ workflow: type })
    );
  });

  it.each([
    ['search', 'search'],
    ['images', 'image'],
    ['videos', 'video'],
    ['news', 'news'],
  ] as const)(
    'always supplies source arrays for the %s workflow callback',
    async (type, bucket) => {
      jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { data: { [bucket]: [row] } },
      });
      const result = await searchAPI().getSources({ query: 'test', type });
      expect(result.success).toBe(true);
      expect(result.data?.organic).toEqual(
        type === 'search' ? [expect.objectContaining({ link: url })] : []
      );
      expect(result.data?.topStories).toEqual([]);
    }
  );

  it('uses the image resource URL when no separate thumbnail exists', async () => {
    jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { data: { image: [{ url, title: 'Image' }] } },
    });
    expect(
      (await searchAPI().getSources({ query: 'test', type: 'images' })).data
        ?.images?.[0].imageUrl
    ).toBe(url);
  });

  it.each([0, 1, 2] as const)(
    'maps safe-search level %s using the documented boolean',
    async (safeSearch) => {
      const post = jest
        .spyOn(axios, 'post')
        .mockResolvedValueOnce({ data: { data: { search: [] } } });
      await searchAPI().getSources({ query: 'test', safeSearch });
      expect(post.mock.calls[0][1]).toEqual(
        expect.objectContaining({ safe_search: safeSearch !== 0 })
      );
    }
  );

  it.each([
    [DATE_RANGE.PAST_HOUR, 'day'],
    [DATE_RANGE.PAST_24_HOURS, 'day'],
    [DATE_RANGE.PAST_WEEK, 'week'],
    [DATE_RANGE.PAST_MONTH, 'month'],
  ])(
    'maps date %s and country with documented filters',
    async (date, interval) => {
      const post = jest
        .spyOn(axios, 'post')
        .mockResolvedValueOnce({ data: { data: { search: [] } } });
      await searchAPI().getSources({ query: 'test', date, country: 'uk' });
      expect(post.mock.calls[0][1]).toEqual(
        expect.objectContaining({
          lens: { time_relative: interval },
          filters: { region: 'GB' },
        })
      );
    }
  );

  it('maps a past-year filter to an ISO calendar date', async () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-10-24T12:30:00Z'));
    try {
      const post = jest
        .spyOn(axios, 'post')
        .mockResolvedValueOnce({ data: { data: { search: [] } } });
      await searchAPI().getSources({
        query: 'test',
        date: DATE_RANGE.PAST_YEAR,
        country: 'de',
      });
      expect(post.mock.calls[0][1]).toEqual(
        expect.objectContaining({
          filters: { region: 'DE', after: '2025-10-24' },
        })
      );
    } finally {
      jest.useRealTimers();
    }
  });

  it.each([
    [0, 1],
    [2048, 1024],
    [3.9, 3],
    [NaN, 8],
    [Infinity, 8],
  ])('bounds limit %s to %s', async (limit, expected) => {
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: { search: [] } } });
    await searchAPI({ limit }).getSources({ query: 'test', numResults: 4 });
    expect(post.mock.calls[0][1]).toEqual(
      expect.objectContaining({ limit: expected })
    );
  });

  it('honors numResults and limits returned rows defensively', async () => {
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: { search: [row, row] } } });
    const result = await searchAPI().getSources({
      query: 'test',
      numResults: 1,
    });
    expect(result.data?.organic).toHaveLength(1);
    expect(post.mock.calls[0][1]).toEqual(
      expect.objectContaining({ limit: 1 })
    );
  });

  it('rejects an empty query without calling HTTP', async () => {
    const post = jest.spyOn(axios, 'post');
    expect(await searchAPI().getSources({ query: '  ' })).toEqual({
      success: false,
      error: 'Query cannot be empty',
    });
    expect(post).not.toHaveBeenCalled();
  });

  it('distinguishes explicit empty success from a malformed response', async () => {
    jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: { search: [] }, error: [] } });
    expect(await searchAPI().getSources({ query: 'test' })).toEqual({
      success: true,
      data: { organic: [], topStories: [] },
    });
  });

  it.each([
    null,
    {},
    { data: {} },
    { data: [] },
    { data: { search: null } },
    { data: { search: {} } },
    { data: { search: [null] } },
    { data: { search: [{ title: 'Missing URL' }] } },
    { data: { search: [{ url, title: 'Bad snippet', snippet: {} }] } },
    { data: { search: [row], image: 'invalid' } },
  ])('rejects malformed search response %#', async (data) => {
    jest.spyOn(axios, 'post').mockResolvedValueOnce({ data });
    expect(await searchAPI().getSources({ query: 'test' })).toEqual({
      success: false,
      error: 'Kagi Search API returned an invalid response',
    });
  });

  it('sanitizes an envelope error even when data is present', async () => {
    jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: {
        data: { search: [row] },
        error: [{ code: secret, message: secret }],
      },
    });
    expect(await searchAPI().getSources({ query: 'test' })).toEqual({
      success: false,
      error: 'Kagi Search API returned an error',
    });
  });

  it.each([401, 429, 500])(
    'does not disclose HTTP %s diagnostics',
    async (status) => {
      jest.spyOn(axios, 'post').mockRejectedValueOnce(
        Object.assign(new Error(secret), {
          config: { headers: { Authorization: `Bearer ${secret}` } },
          response: { status, data: { error: secret } },
        })
      );
      expect(await searchAPI().getSources({ query: 'test' })).toEqual({
        success: false,
        error: 'Kagi Search API request failed',
      });
    }
  );

  it('resolves global transport agents ahead of provider options', async () => {
    const httpAgent = new HttpAgent();
    const httpsAgent = new HttpsAgent();
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValue({ data: { data: { search: [] } } });
    await createSearchAPI({
      searchProvider: 'kagi',
      kagiApiKey: 'key',
      httpAgent,
      httpsAgent,
      kagiSearchOptions: {
        httpAgent: new HttpAgent(),
        httpsAgent: new HttpsAgent(),
        timeout: 1234,
      },
    }).getSources({ query: 'test' });
    expect(post.mock.calls[0][2]).toEqual(
      expect.objectContaining({ httpAgent, httpsAgent, timeout: 1234 })
    );
    await searchAPI({ httpAgent, httpsAgent }).getSources({ query: 'test' });
    expect(post.mock.calls[1][2]).toEqual(
      expect.objectContaining({ httpAgent, httpsAgent })
    );
  });
});

describe('Kagi Extract v1', () => {
  it('posts pages and returns Markdown without interpreting it as HTML', async () => {
    const markdown =
      '# Heading\n\n**Evidence** [source](https://example.com)\n\n<literal>';
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: [{ url, markdown }] } });
    const scraper = createKagiScraper({ apiKey: 'extract-key' });
    const [link, result] = await scraper.scrapeUrl(url);
    expect(post).toHaveBeenCalledWith(
      extractUrl,
      { pages: [{ url }], format: 'json' },
      expect.objectContaining({
        headers: {
          Authorization: 'Bearer extract-key',
          'Content-Type': 'application/json',
        },
        timeout: 7500,
        maxRedirects: 0,
      })
    );
    expect(link).toBe(url);
    expect(result).toEqual({ success: true, data: { url, markdown } });
    expect(scraper.extractContent(result)).toEqual([markdown, undefined]);
    expect(scraper.extractMetadata(result)).toEqual({ url });
  });

  it('keeps partial successes in input order and sanitizes page and plural errors', async () => {
    const missing = 'https://example.com/missing';
    const failed = 'https://example.com/failed';
    jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: {
        data: [
          { url: failed, markdown: '# Discard this', error: secret },
          { url, markdown: '# Good page' },
        ],
        errors: [{ code: secret, message: secret, location: 'pages[2].url' }],
      },
    });
    const scraper = createKagiScraper({ apiKey: 'key' });
    const results = await scraper.scrapeUrls([url, failed, missing]);
    expect(results).toEqual([
      [url, { success: true, data: { url, markdown: '# Good page' } }],
      [
        failed,
        {
          success: false,
          error: 'Kagi Extract could not extract page content',
        },
      ],
      [
        missing,
        {
          success: false,
          error: 'Kagi Extract could not extract page content',
        },
      ],
    ]);
    expect(scraper.extractContent(results[1][1])).toEqual(['', undefined]);
    expect(scraper.extractMetadata(results[1][1])).toEqual({});
    expect(JSON.stringify(results)).not.toContain(secret);
  });

  it.each([null, undefined, '', '   '])(
    'treats absent/empty Markdown %p as a per-page failure',
    async (markdown) => {
      jest
        .spyOn(axios, 'post')
        .mockResolvedValueOnce({ data: { data: [{ url, markdown }] } });
      expect(
        (await createKagiScraper({ apiKey: 'key' }).scrapeUrl(url))[1].success
      ).toBe(false);
    }
  );

  it.each([
    null,
    {},
    { data: null },
    { data: {} },
    { data: [null] },
    { data: [] },
  ])('fails safely on malformed/missing extract response %#', async (data) => {
    jest.spyOn(axios, 'post').mockResolvedValueOnce({ data });
    const result = await createKagiScraper({ apiKey: 'key' }).scrapeUrl(url);
    expect(result[1].success).toBe(false);
    expect(result[1].error).toMatch(/^Kagi Extract /);
  });

  it('handles a singular envelope error without forwarding data or diagnostics', async () => {
    jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: {
        data: [{ url, markdown: '# Ignore me' }],
        error: [{ code: secret, message: secret }],
      },
    });
    expect(
      (await createKagiScraper({ apiKey: 'key' }).scrapeUrl(url))[1]
    ).toEqual({ success: false, error: 'Kagi Extract API returned an error' });
  });

  it('sanitizes network failures', async () => {
    jest.spyOn(axios, 'post').mockRejectedValueOnce(new Error(secret));
    expect(
      (await createKagiScraper({ apiKey: 'key' }).scrapeUrl(url))[1]
    ).toEqual({ success: false, error: 'Kagi Extract API request failed' });
  });

  it('splits batches at ten, preserves duplicates/order, and isolates batch failures', async () => {
    const urls = Array.from(
      { length: 12 },
      (_, index) => `https://example.com/${index}`
    );
    urls.push(urls[0]);
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({
        data: {
          data: urls
            .slice(0, 10)
            .reverse()
            .map((link) => ({ url: link, markdown: '# Page' })),
        },
      })
      .mockRejectedValueOnce(new Error(secret));
    const results = await createKagiScraper({ apiKey: 'key' }).scrapeUrls(urls);
    expect(post).toHaveBeenCalledTimes(2);
    expect(post.mock.calls[0][1]).toEqual({
      pages: urls.slice(0, 10).map((link) => ({ url: link })),
      format: 'json',
    });
    expect(post.mock.calls[1][1]).toEqual({
      pages: urls.slice(10).map((link) => ({ url: link })),
      format: 'json',
    });
    expect(results.map(([link]) => link)).toEqual(urls);
    expect(results.slice(0, 10).every(([, result]) => result.success)).toBe(
      true
    );
    expect(results.slice(10).every(([, result]) => !result.success)).toBe(true);
  });

  it('handles empty batches and invalid URLs without HTTP', async () => {
    const post = jest.spyOn(axios, 'post');
    const scraper = createKagiScraper({ apiKey: 'key' });
    expect(await scraper.scrapeUrls([])).toEqual([]);
    const results = await scraper.scrapeUrls(['', 'file:///etc/passwd']);
    expect(results.every(([, result]) => !result.success)).toBe(true);
    expect(post).not.toHaveBeenCalled();
  });

  it('allows a per-call HTTP timeout override and passes configured agents', async () => {
    const httpAgent = new HttpAgent();
    const httpsAgent = new HttpsAgent();
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: [{ url, markdown: '# Page' }] } });
    await createKagiScraper({
      apiKey: 'key',
      timeout: 2000,
      httpAgent,
      httpsAgent,
    }).scrapeUrl(url, { timeout: 3000 });
    expect(post.mock.calls[0][2]).toEqual(
      expect.objectContaining({ timeout: 3000, httpAgent, httpsAgent })
    );
    expect(post.mock.calls[0][1]).not.toHaveProperty('timeout');
  });
});

describe('Kagi tool pipeline', () => {
  it('shares the key, preserves artifacts/citations, reranking, media searches and metrics', async () => {
    const post = jest
      .spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { data: { search: [row] } } })
      .mockResolvedValueOnce({ data: { data: { image: [row] } } })
      .mockResolvedValueOnce({ data: { data: { video: [row] } } })
      .mockResolvedValueOnce({ data: { data: { news: [] } } })
      .mockResolvedValueOnce({
        data: { data: [{ url, markdown: '# Extracted evidence' }] },
      });
    const onSearchResults = jest.fn();
    const debug = jest.spyOn(logger, 'debug');
    const rerank = jest.fn(
      async (_query: string, documents: string[]): Promise<t.Highlight[]> => [
        { text: documents[0], score: 1 },
      ]
    );
    const tool = createSearchTool({
      searchProvider: 'kagi',
      scraperProvider: 'kagi',
      kagiApiKey: 'shared-key',
      kagiSearchOptions: { limit: 2, timeout: 4321 },
      safeSearch: 0,
      reranker: { provider: 'host', rerank },
      rerankerType: 'none',
      topResults: 1,
      logger,
      onSearchResults,
    });
    expect(tool.schema).toEqual(
      expect.objectContaining({
        properties: expect.objectContaining({ country: expect.anything() }),
      })
    );
    const message = await tool.invoke({
      type: 'tool_call',
      id: 'kagi-call',
      name: 'web_search',
      args: {
        query: 'evidence',
        country: 'ca',
        images: true,
        videos: true,
        news: true,
      },
    });
    expect(post).toHaveBeenCalledTimes(5);
    for (const call of post.mock.calls) {
      expect(call[2]?.headers?.Authorization).toBe('Bearer shared-key');
    }
    expect(post.mock.calls[0][1]).toEqual(
      expect.objectContaining({
        limit: 2,
        safe_search: false,
        filters: { region: 'CA' },
      })
    );
    expect(post.mock.calls[0][2]?.timeout).toBe(4321);
    expect(post.mock.calls[4][0]).toBe(extractUrl);
    expect(rerank).toHaveBeenCalledWith(
      'evidence',
      expect.arrayContaining(['# Extracted evidence']),
      1,
      expect.anything()
    );
    expect(JSON.stringify(message)).toContain('Extracted evidence');
    expect(JSON.stringify(message)).toContain('turn0search0');
    expect(message.artifact).toEqual(
      expect.objectContaining({
        web_search: expect.objectContaining({
          organic: [expect.objectContaining({ link: url })],
          images: [expect.objectContaining({ link: url })],
          videos: [expect.objectContaining({ link: url })],
        }),
      })
    );
    expect(onSearchResults).toHaveBeenCalled();
    expect(debug.mock.calls.flat().join(' ')).toContain(
      'search=kagi queries=4'
    );
    expect(debug.mock.calls.flat().join(' ')).toContain('scrape links=1 ok=1');
  });

  it.each([undefined, 4567])(
    'resolves scraper timeout %p ahead of provider options and honors scraper key override',
    async (scraperTimeout) => {
      const httpAgent = new HttpAgent();
      const httpsAgent = new HttpsAgent();
      const post = jest
        .spyOn(axios, 'post')
        .mockResolvedValueOnce({ data: { data: { search: [row] } } })
        .mockResolvedValueOnce({
          data: { data: [{ url, markdown: '# Evidence' }] },
        });
      await createSearchTool({
        searchProvider: 'kagi',
        scraperProvider: 'kagi',
        kagiApiKey: 'shared-key',
        kagiScraperOptions: {
          apiKey: 'extract-key',
          timeout: 1234,
          httpAgent: new HttpAgent(),
          httpsAgent: new HttpsAgent(),
        },
        scraperTimeout,
        httpAgent,
        httpsAgent,
        rerankerType: 'none',
        topResults: 1,
        logger,
      }).invoke({ query: 'test' });
      expect(post.mock.calls[1][2]).toEqual(
        expect.objectContaining({
          timeout: scraperTimeout ?? 1234,
          httpAgent,
          httpsAgent,
          headers: expect.objectContaining({
            Authorization: 'Bearer extract-key',
          }),
        })
      );
    }
  );

  it.each(['search', 'extract'])(
    'mixes Kagi %s with Serper independently',
    async (mode) => {
      const search = mode === 'search';
      const post = jest
        .spyOn(axios, 'post')
        .mockResolvedValueOnce({
          data: search
            ? { data: { search: [row] } }
            : { organic: [{ link: url, title: 'Evidence' }] },
        })
        .mockResolvedValueOnce({
          data: search
            ? { text: 'Extracted evidence' }
            : { data: [{ url, markdown: '# Evidence' }] },
        });
      const result = await createSearchTool({
        searchProvider: search ? 'kagi' : 'serper',
        scraperProvider: search ? 'serper' : 'kagi',
        kagiApiKey: 'kagi-key',
        serperApiKey: 'serper-key',
        rerankerType: 'none',
        topResults: 1,
        logger,
      }).invoke({ query: 'test' });
      expect(post).toHaveBeenCalledTimes(2);
      expect(post.mock.calls[search ? 0 : 1][0]).toBe(
        search ? searchUrl : extractUrl
      );
      expect(JSON.stringify(result)).toContain('Evidence');
    }
  );

  it('keeps HTTP errors out of tool output, artifacts, callbacks and logs', async () => {
    jest.spyOn(axios, 'post').mockRejectedValueOnce(new Error(secret));
    const error = jest.spyOn(logger, 'error');
    const warn = jest.spyOn(logger, 'warn');
    const onSearchResults = jest.fn();
    const message = await createSearchTool({
      searchProvider: 'kagi',
      scraperProvider: 'kagi',
      kagiApiKey: 'key',
      rerankerType: 'none',
      logger,
      onSearchResults,
    }).invoke({
      type: 'tool_call',
      id: 'failed-call',
      name: 'web_search',
      args: { query: 'test' },
    });
    const serialized = JSON.stringify([
      message,
      onSearchResults.mock.calls,
      error.mock.calls,
      warn.mock.calls,
    ]);
    expect(serialized).toContain('Kagi Search API request failed');
    expect(serialized).not.toContain(secret);
  });
});

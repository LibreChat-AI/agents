import { createServer } from 'http';
import type { AddressInfo } from 'net';
import { createFirecrawlScraper } from './firecrawl';
import { createDefaultLogger } from './utils';

interface CapturedRequest {
  path?: string;
  authorization?: string;
}

describe('FirecrawlScraper authorization', () => {
  const requests: CapturedRequest[] = [];
  const server = createServer((req, res) => {
    requests.push({ path: req.url, authorization: req.headers.authorization });
    req.resume();
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      res.end(
        JSON.stringify({
          success: true,
          data: { markdown: '# Self-hosted page' },
        })
      );
    });
  });
  const logger = createDefaultLogger();
  const originalKey = process.env.FIRECRAWL_API_KEY;
  let apiUrl = '';

  beforeAll(async () => {
    await new Promise<void>((resolve) =>
      server.listen(0, '127.0.0.1', resolve)
    );
    const { port } = server.address() as AddressInfo;
    apiUrl = `http://127.0.0.1:${port}`;
  });

  afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  beforeEach(() => {
    requests.length = 0;
    delete process.env.FIRECRAWL_API_KEY;
    jest.spyOn(logger, 'warn').mockImplementation(() => logger);
    jest.spyOn(logger, 'debug').mockImplementation(() => logger);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    if (originalKey === undefined) {
      delete process.env.FIRECRAWL_API_KEY;
    } else {
      process.env.FIRECRAWL_API_KEY = originalKey;
    }
  });

  it('scrapes without an API key and sends no Authorization header', async () => {
    const scraper = createFirecrawlScraper({ apiUrl, logger });
    const [url, response] = await scraper.scrapeUrl('https://example.com/page');

    expect(url).toBe('https://example.com/page');
    expect(response.success).toBe(true);
    expect(scraper.extractContent(response)[0]).toBe('# Self-hosted page');
    expect(requests).toEqual([
      { path: '/v2/scrape', authorization: undefined },
    ]);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('treats an empty key as no key even when FIRECRAWL_API_KEY is set', async () => {
    process.env.FIRECRAWL_API_KEY = 'environment-key';
    const scraper = createFirecrawlScraper({ apiKey: '', apiUrl, logger });
    await scraper.scrapeUrl('https://example.com/page');

    expect(requests).toEqual([
      { path: '/v2/scrape', authorization: undefined },
    ]);
  });

  it('sends a configured key as a bearer token', async () => {
    const scraper = createFirecrawlScraper({
      apiKey: 'fc-key',
      apiUrl,
      version: 'v1',
      logger,
    });
    await scraper.scrapeUrl('https://example.com/page');

    expect(requests).toEqual([
      { path: '/v1/scrape', authorization: 'Bearer fc-key' },
    ]);
  });
});

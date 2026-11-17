import axios from 'axios';

import { createReranker, JinaReranker } from './rerankers';
import { createDefaultLogger } from './utils';

type AxiosErrorFixture = Error & {
  isAxiosError: true;
  code: string;
  config: {
    data: string;
    headers: {
      Authorization: string;
    };
    method: string;
    url: string;
  };
  response: {
    data: {
      details?: string;
      message: string;
    };
    status: number;
  };
};

const getApiUrl = (reranker: JinaReranker): string => {
  const descriptor = Object.getOwnPropertyDescriptor(reranker, 'apiUrl');
  if (typeof descriptor?.value === 'string') {
    return descriptor.value;
  }
  throw new Error('Expected JinaReranker apiUrl to be initialized.');
};

const createAxiosErrorFixture = (
  url: string,
  responseData: AxiosErrorFixture['response']['data']
): AxiosErrorFixture =>
  Object.assign(new Error('Request failed with status code 500'), {
    isAxiosError: true as const,
    code: 'ERR_BAD_RESPONSE',
    config: {
      data: JSON.stringify({ documents: ['document1', 'document2'] }),
      headers: {
        Authorization: 'Bearer test-key',
      },
      method: 'post',
      url,
    },
    response: {
      data: responseData,
      status: 500,
    },
  });

describe('JinaReranker', () => {
  const mockLogger = createDefaultLogger();

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should use default API URL when no apiUrl is provided', () => {
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      const apiUrl = getApiUrl(reranker);
      expect(apiUrl).toBe('https://api.jina.ai/v1/rerank');
    });

    it('should use custom API URL when provided', () => {
      const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const apiUrl = getApiUrl(reranker);
      expect(apiUrl).toBe(customUrl);
    });

    it('should use environment variable JINA_API_URL when available', () => {
      const originalEnv = process.env.JINA_API_URL;
      process.env.JINA_API_URL = 'https://env-jina-endpoint.com/v1/rerank';

      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      const apiUrl = getApiUrl(reranker);
      expect(apiUrl).toBe('https://env-jina-endpoint.com/v1/rerank');

      if (typeof originalEnv === 'string') {
        process.env.JINA_API_URL = originalEnv;
      } else {
        delete process.env.JINA_API_URL;
      }
    });

    it('should prioritize explicit apiUrl over environment variable', () => {
      const originalEnv = process.env.JINA_API_URL;
      process.env.JINA_API_URL = 'https://env-jina-endpoint.com/v1/rerank';

      const customUrl = 'https://explicit-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const apiUrl = getApiUrl(reranker);
      expect(apiUrl).toBe(customUrl);

      if (typeof originalEnv === 'string') {
        process.env.JINA_API_URL = originalEnv;
      } else {
        delete process.env.JINA_API_URL;
      }
    });
  });

  describe('rerank method', () => {
    it('should post to the configured API URL', async () => {
      const customUrl = 'https://test-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      jest.spyOn(mockLogger, 'error').mockImplementation(() => mockLogger);
      const postSpy = jest
        .spyOn(axios, 'post')
        .mockRejectedValueOnce(new Error('Network error'));

      await reranker.rerank('test query', ['document1', 'document2'], 2);

      expect(postSpy).toHaveBeenCalledWith(
        customUrl,
        expect.objectContaining({ top_n: 2 }),
        expect.anything()
      );
    });

    it('should send configured headers through createReranker', async () => {
      const reranker = createReranker({
        rerankerType: 'jina',
        jinaApiKey: 'test-key',
        rerankerHeaders: { 'x-end-user': 'user@example.com' },
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      jest.spyOn(mockLogger, 'error').mockImplementation(() => mockLogger);
      const postSpy = jest
        .spyOn(axios, 'post')
        .mockRejectedValueOnce(new Error('Network error'));

      await reranker?.rerank('test query', ['document1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        expect.any(String),
        expect.anything(),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-key',
            'x-end-user': 'user@example.com',
          }),
        })
      );
    });

    it('should keep a ranking when the response omits the usage telemetry', async () => {
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const warnSpy = jest
        .spyOn(mockLogger, 'warn')
        .mockImplementation(() => mockLogger);
      const errorSpy = jest
        .spyOn(mockLogger, 'error')
        .mockImplementation(() => mockLogger);
      // A Jina-compatible endpoint may return results with no `usage` block.
      jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          model: 'custom-reranker',
          results: [
            { index: 1, relevance_score: 0.9 },
            { index: 0, relevance_score: 0.4 },
          ],
        },
      });

      const result = await reranker.rerank(
        'test query',
        ['document1', 'document2'],
        2
      );

      expect(result).toEqual([
        { text: 'document2', score: 0.9 },
        { text: 'document1', score: 0.4 },
      ]);
      expect(warnSpy).not.toHaveBeenCalled();
      expect(errorSpy).not.toHaveBeenCalled();
    });

    it('should log compact Axios errors without request internals', async () => {
      const customUrl =
        'https://test-jina-endpoint.com/v1/rerank?api_key=hidden';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });
      const errorSpy = jest
        .spyOn(mockLogger, 'error')
        .mockImplementation(() => mockLogger);
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);
      jest.spyOn(axios, 'post').mockRejectedValueOnce(
        createAxiosErrorFixture(customUrl, {
          message: 'upstream failed',
          details: 'x'.repeat(5000),
        })
      );

      const result = await reranker.rerank(
        'test query',
        ['document1', 'document2'],
        2
      );

      expect(result).toEqual([
        { text: 'document1', score: 0 },
        { text: 'document2', score: 0 },
      ]);
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining('rerank=jina calls=1'),
        expect.objectContaining({
          code: 'ERR_BAD_RESPONSE',
          message: 'Request failed with status code 500',
          method: 'POST',
          name: 'Error',
          responseDataSummary: 'object(2)',
          status: 500,
          url: 'https://test-jina-endpoint.com/v1/rerank',
        })
      );

      const metadata = errorSpy.mock.calls.flat()[1];
      const serializedMetadata = JSON.stringify(metadata);

      expect(serializedMetadata).not.toContain('upstream failed');
      expect(serializedMetadata).not.toContain('Authorization');
      expect(serializedMetadata).not.toContain('api_key');
      expect(serializedMetadata).not.toContain('document1');
      expect(serializedMetadata).not.toContain('test-key');
      expect(serializedMetadata.length).toBeLessThan(500);
    });

    it('should bound the rerank request with the default timeout', async () => {
      const customUrl = 'https://test-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ index: 0, relevance_score: 0.9 }] },
      });

      await reranker.rerank('test query', ['document1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        customUrl,
        expect.any(Object),
        expect.objectContaining({ timeout: 10000 })
      );
    });

    it('should bound the rerank request with a custom timeout', async () => {
      const customUrl = 'https://test-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        timeout: 2500,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ index: 0, relevance_score: 0.9 }] },
      });

      await reranker.rerank('test query', ['document1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        customUrl,
        expect.any(Object),
        expect.objectContaining({ timeout: 2500 })
      );
    });
  });
});

describe('createReranker', () => {
  it('should create JinaReranker with jinaApiUrl when provided', () => {
    const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
      jinaApiUrl: customUrl,
    });

    expect(reranker).toBeInstanceOf(JinaReranker);
    if (!(reranker instanceof JinaReranker)) {
      throw new Error('Expected createReranker to return a JinaReranker.');
    }
    const apiUrl = getApiUrl(reranker);
    expect(apiUrl).toBe(customUrl);
  });

  it('should create JinaReranker with default URL when jinaApiUrl is not provided', () => {
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
    });

    expect(reranker).toBeInstanceOf(JinaReranker);
    if (!(reranker instanceof JinaReranker)) {
      throw new Error('Expected createReranker to return a JinaReranker.');
    }
    const apiUrl = getApiUrl(reranker);
    expect(apiUrl).toBe('https://api.jina.ai/v1/rerank');
  });

  it('should pass rerankerTimeout through to the rerank request', async () => {
    const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
      jinaApiUrl: customUrl,
      rerankerTimeout: 4000,
      logger: createDefaultLogger(),
    });
    if (!(reranker instanceof JinaReranker)) {
      throw new Error('Expected createReranker to return a JinaReranker.');
    }
    const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { results: [{ index: 0, relevance_score: 0.9 }] },
    });

    await reranker.rerank('test query', ['document1'], 1);

    expect(postSpy).toHaveBeenCalledWith(
      customUrl,
      expect.any(Object),
      expect.objectContaining({ timeout: 4000 })
    );
  });
});

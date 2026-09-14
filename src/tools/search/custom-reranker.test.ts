/* eslint-disable @typescript-eslint/no-explicit-any */
import axios from 'axios';
import { CustomReranker, createReranker } from './rerankers';
import { createDefaultLogger } from './utils';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('CustomReranker', () => {
  const mockLogger = createDefaultLogger();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should use environment variables as fallbacks', () => {
      const originalUrl = process.env.CUSTOM_RERANKER_API_URL;
      const originalKey = process.env.CUSTOM_RERANKER_API_KEY;
      const originalModel = process.env.CUSTOM_RERANKER_MODEL;

      process.env.CUSTOM_RERANKER_API_URL =
        'https://env-endpoint.com/v1/rerank';
      process.env.CUSTOM_RERANKER_API_KEY = 'env-key';
      process.env.CUSTOM_RERANKER_MODEL = 'env-model';

      const reranker = new CustomReranker({ logger: mockLogger });

      expect((reranker as any).apiUrl).toBe(
        'https://env-endpoint.com/v1/rerank'
      );
      expect((reranker as any).apiKey).toBe('env-key');
      expect((reranker as any).model).toBe('env-model');

      if (originalUrl !== undefined) {
        process.env.CUSTOM_RERANKER_API_URL = originalUrl;
      } else {
        delete process.env.CUSTOM_RERANKER_API_URL;
      }
      if (originalKey !== undefined) {
        process.env.CUSTOM_RERANKER_API_KEY = originalKey;
      } else {
        delete process.env.CUSTOM_RERANKER_API_KEY;
      }
      if (originalModel !== undefined) {
        process.env.CUSTOM_RERANKER_MODEL = originalModel;
      } else {
        delete process.env.CUSTOM_RERANKER_MODEL;
      }
    });

    it('should prioritize explicit params over environment variables', () => {
      const originalUrl = process.env.CUSTOM_RERANKER_API_URL;
      process.env.CUSTOM_RERANKER_API_URL =
        'https://env-endpoint.com/v1/rerank';

      const reranker = new CustomReranker({
        apiUrl: 'https://explicit-endpoint.com/v1/rerank',
        apiKey: 'explicit-key',
        model: 'explicit-model',
        logger: mockLogger,
      });

      expect((reranker as any).apiUrl).toBe(
        'https://explicit-endpoint.com/v1/rerank'
      );
      expect((reranker as any).apiKey).toBe('explicit-key');
      expect((reranker as any).model).toBe('explicit-model');

      if (originalUrl !== undefined) {
        process.env.CUSTOM_RERANKER_API_URL = originalUrl;
      } else {
        delete process.env.CUSTOM_RERANKER_API_URL;
      }
    });
  });

  describe('rerank', () => {
    it('should fall back to default ranking when apiUrl is missing', async () => {
      const reranker = new CustomReranker({
        model: 'test-model',
        logger: mockLogger,
      });

      const result = await reranker.rerank('query', ['doc1', 'doc2'], 2);

      expect(result).toEqual([
        { text: 'doc1', score: 0 },
        { text: 'doc2', score: 0 },
      ]);
      expect(mockedAxios.post).not.toHaveBeenCalled();
    });

    it('should fall back to default ranking when model is missing', async () => {
      const reranker = new CustomReranker({
        apiUrl: 'https://example.com/v1/rerank',
        logger: mockLogger,
      });

      const result = await reranker.rerank('query', ['doc1', 'doc2'], 2);

      expect(result).toEqual([
        { text: 'doc1', score: 0 },
        { text: 'doc2', score: 0 },
      ]);
      expect(mockedAxios.post).not.toHaveBeenCalled();
    });

    it('should send request without Authorization header when apiKey is missing', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          model: 'test-model',
          usage: { total_tokens: 10 },
          results: [
            { index: 0, relevance_score: 0.9, document: { text: 'doc1' } },
          ],
        },
      });

      const reranker = new CustomReranker({
        apiUrl: 'https://example.com/v1/rerank',
        model: 'test-model',
        logger: mockLogger,
      });

      await reranker.rerank('query', ['doc1', 'doc2'], 1);

      expect(mockedAxios.post).toHaveBeenCalledWith(
        'https://example.com/v1/rerank',
        expect.any(Object),
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: expect.any(Number),
        }
      );
    });

    it('should send correct request with custom model name and auth', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          model: 'qwen3-embedding-8b',
          usage: { total_tokens: 50 },
          results: [
            { index: 1, relevance_score: 0.95, document: { text: 'doc2' } },
            { index: 0, relevance_score: 0.8, document: { text: 'doc1' } },
          ],
        },
      });

      const reranker = new CustomReranker({
        apiUrl: 'https://api.scaleway.ai/v1/rerank',
        apiKey: 'my-key',
        model: 'qwen3-embedding-8b',
        timeout: 4321,
        logger: mockLogger,
      });

      const result = await reranker.rerank('test query', ['doc1', 'doc2'], 2);

      expect(mockedAxios.post).toHaveBeenCalledWith(
        'https://api.scaleway.ai/v1/rerank',
        {
          model: 'qwen3-embedding-8b',
          query: 'test query',
          top_n: 2,
          documents: ['doc1', 'doc2'],
          return_documents: true,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer my-key',
          },
          timeout: 4321,
        }
      );

      expect(result).toEqual([
        { text: 'doc2', score: 0.95 },
        { text: 'doc1', score: 0.8 },
      ]);
    });

    it('should handle API errors gracefully', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

      const reranker = new CustomReranker({
        apiUrl: 'https://example.com/v1/rerank',
        apiKey: 'key',
        model: 'model',
        logger: mockLogger,
      });

      const result = await reranker.rerank('query', ['doc1', 'doc2'], 2);

      expect(result).toEqual([
        { text: 'doc1', score: 0 },
        { text: 'doc2', score: 0 },
      ]);
    });

    it('should handle document as string in response', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          model: 'test-model',
          usage: { total_tokens: 10 },
          results: [{ index: 0, relevance_score: 0.9, document: 'doc1-text' }],
        },
      });

      const reranker = new CustomReranker({
        apiUrl: 'https://example.com/v1/rerank',
        model: 'test-model',
        logger: mockLogger,
      });

      const result = await reranker.rerank('query', ['doc1'], 1);

      expect(result).toEqual([{ text: 'doc1-text', score: 0.9 }]);
    });

    it('should use document index when document field is missing', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          model: 'test-model',
          usage: { total_tokens: 10 },
          results: [{ index: 1, relevance_score: 0.8 }],
        },
      });

      const reranker = new CustomReranker({
        apiUrl: 'https://example.com/v1/rerank',
        model: 'test-model',
        logger: mockLogger,
      });

      const result = await reranker.rerank('query', ['doc1', 'doc2'], 1);

      expect(result).toEqual([{ text: 'doc2', score: 0.8 }]);
    });
  });
});

describe('createReranker with custom type', () => {
  it('should create CustomReranker when rerankerType is custom', () => {
    const reranker = createReranker({
      rerankerType: 'custom',
      customRerankerApiUrl: 'https://example.com/v1/rerank',
      customRerankerApiKey: 'key',
      customRerankerModel: 'model',
    });

    expect(reranker).toBeInstanceOf(CustomReranker);
    expect((reranker as any).apiUrl).toBe('https://example.com/v1/rerank');
    expect((reranker as any).apiKey).toBe('key');
    expect((reranker as any).model).toBe('model');
  });
});

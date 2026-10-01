import axios from 'axios';
import { createLogger } from 'winston';
import type { SearchMetrics, SearchReranker } from './types';
import { createSearchTool } from './tool';

const logger = createLogger({ silent: true });
const content = 'FIRST EVIDENCE.\nSECOND EVIDENCE.\nTHIRD EVIDENCE.';

function providerResponses(): jest.SpyInstance {
  return jest
    .spyOn(axios, 'post')
    .mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          web: [
            {
              title: 'Evidence',
              url: 'https://example.com/evidence',
              description: 'Snippet',
            },
          ],
        },
      },
    })
    .mockResolvedValueOnce({
      data: { success: true, data: { markdown: content } },
    });
}

afterEach(() => jest.restoreAllMocks());

describe('host-injected web-search reranker', () => {
  it.each(['none', 'cohere', 'jina'] as const)(
    'honors the injection before built-in %s selection',
    async (rerankerType) => {
      const post = providerResponses();
      const rerank = jest.fn(
        async (
          _query: string,
          documents: string[],
          _topK?: number,
          metrics?: SearchMetrics
        ) => {
          metrics?.recordRerank({
            provider: 'host',
            chunks: documents.length,
            results: 1,
            durationMs: 0,
          });
          return [{ text: documents[documents.length - 1], score: 0.75 }];
        }
      );
      const reranker: SearchReranker = { provider: 'host', rerank };
      const tool = createSearchTool({
        searchProvider: 'crw',
        scraperProvider: 'crw',
        crwApiKey: 'test',
        reranker,
        rerankerType,
        topResults: 1,
        chunkSize: 20,
        chunkOverlap: 0,
        logger,
      });
      await tool.invoke({ query: 'evidence' });
      expect(rerank).toHaveBeenCalledTimes(1);
      expect(rerank).toHaveBeenCalledWith(
        'evidence',
        expect.arrayContaining(['FIRST EVIDENCE.']),
        1,
        expect.objectContaining({ recordRerank: expect.any(Function) })
      );
      expect(post).toHaveBeenCalledTimes(2);
    }
  );

  it('keeps built-in Cohere selection when no host reranker is supplied', async () => {
    const post = providerResponses().mockResolvedValueOnce({
      data: { results: [{ index: 0, relevance_score: 0.8 }] },
    });
    const tool = createSearchTool({
      searchProvider: 'crw',
      scraperProvider: 'crw',
      crwApiKey: 'test',
      rerankerType: 'cohere',
      cohereApiKey: 'test',
      topResults: 1,
      logger,
    });
    await tool.invoke({ query: 'evidence' });
    expect(post).toHaveBeenCalledTimes(3);
    expect(post.mock.calls[2][0]).toBe('https://api.cohere.com/v2/rerank');
  });
});

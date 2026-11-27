import { concat } from '@langchain/core/utils/stream';
import { HumanMessage } from '@langchain/core/messages';
import { expect, describe, it, jest } from '@jest/globals';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { VertexAIClientOptions } from '@/types';
import { ChatVertexAI } from './index';

type GeminiChunk = Record<string, unknown>;

function makeJsonStream(
  chunks: GeminiChunk[],
  withTail = true,
  onEnd?: () => void
) {
  let index = 0;
  let ended = false;
  return {
    get streamDone() {
      const done = withTail ? index > chunks.length : index >= chunks.length;
      if (done && !ended) {
        ended = true;
        onEnd?.();
      }
      return done;
    },
    async nextChunk(): Promise<GeminiChunk | null> {
      if (index >= chunks.length) {
        index += 1;
        return null;
      }
      return chunks[index++];
    },
  };
}

function modelOutput(text: string, extra: GeminiChunk = {}): GeminiChunk {
  const { finishReason, continuationToken, ...rest } = extra;
  return {
    candidates: [
      {
        content: { role: 'model', parts: [{ text }] },
        finishReason,
        continuationToken,
      },
    ],
    ...rest,
  };
}

function vertex(fields: VertexAIClientOptions): ChatVertexAI {
  return new ChatVertexAI({
    authOptions: {
      projectId: 'test-project',
      credentials: { client_email: 'test@test', private_key: 'test' },
    },
    ...fields,
  });
}

function apiVersion(fields: VertexAIClientOptions): string {
  const model = vertex(fields) as unknown as {
    connection: { apiVersion: string };
  };
  return model.connection.apiVersion;
}

function createModel(
  segments: GeminiChunk[][],
  fields: VertexAIClientOptions = {},
  withTail = true,
  onSegmentEnd?: (segment: number) => void
) {
  const model = vertex({
    model: 'gemini-4-carbon',
    maxOutputTokens: 131072,
    ...fields,
  });
  const { streamedConnection } = model as unknown as {
    streamedConnection: {
      request: unknown;
      formatData: (
        input: unknown,
        parameters: Record<string, unknown>
      ) => Promise<Record<string, unknown>>;
    };
  };
  const bodies: Record<string, unknown>[] = [];
  streamedConnection.request = jest.fn(
    async (
      input: unknown,
      parameters: Record<string, unknown>
    ): Promise<{ data: ReturnType<typeof makeJsonStream> }> => {
      const segment = bodies.length;
      bodies.push({});
      bodies[segment] = await streamedConnection.formatData(input, parameters);
      return {
        data: makeJsonStream(segments[segment], withTail, () =>
          onSegmentEnd?.(segment)
        ),
      };
    }
  );
  return { model, bodies };
}

async function collectMessages(
  model: ChatVertexAI,
  options: { signal?: AbortSignal } = {}
): Promise<AIMessageChunk[]> {
  const chunks: AIMessageChunk[] = [];
  const stream = await model.stream([new HumanMessage('hi')], {
    signal: options.signal,
  });
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return chunks;
}

function combineMessages(chunks: AIMessageChunk[]): AIMessageChunk {
  return chunks.slice(1).reduce((sum, chunk) => concat(sum, chunk), chunks[0]);
}

const usage = (
  prompt: number,
  candidates: number,
  thoughts: number
): GeminiChunk => ({
  usageMetadata: {
    promptTokenCount: prompt,
    candidatesTokenCount: candidates,
    thoughtsTokenCount: thoughts,
    totalTokenCount: prompt + candidates + thoughts,
  },
});

describe('Gemini 4 Carbon long decode', () => {
  it('continues with the original request plus continuationToken and aggregates usage', async () => {
    const { model, bodies } = createModel([
      [
        modelOutput('reasoning', {
          finishReason: 'CONTINUATION',
          continuationToken: 'segment-2',
          ...usage(100, 0, 65533),
        }),
      ],
      [
        modelOutput('final answer', {
          finishReason: 'STOP',
          ...usage(200, 5, 0),
        }),
      ],
    ]);

    const chunks = await collectMessages(model);

    expect(bodies).toHaveLength(2);
    expect(bodies[1]).toEqual({ ...bodies[0], continuationToken: 'segment-2' });
    expect(bodies[1].generationConfig).toEqual(bodies[0].generationConfig);
    expect(bodies[1].contents).toEqual(bodies[0].contents);
    expect(bodies[1].systemInstruction).toEqual(bodies[0].systemInstruction);
    expect(bodies[1].tools).toEqual(bodies[0].tools);

    const finishReasons = chunks.map(
      (chunk) => chunk.response_metadata.finish_reason
    );
    expect(finishReasons).not.toContain('CONTINUATION');
    expect(finishReasons).toContain('STOP');

    const full = combineMessages(chunks);
    expect(full.usage_metadata).toMatchObject({
      input_tokens: 300,
      output_tokens: 65538,
      total_tokens: 65838,
      output_token_details: { reasoning: 65533 },
    });
    expect(full.response_metadata.initial_prompt_token_count).toBe(100);
    expect(full.response_metadata.continuation_token).toBeUndefined();
    expect(full.response_metadata.finishReason).toBe('stop');
  });

  it('passes MAX_TOKENS through without continuing or raising the configured budget', async () => {
    const { model, bodies } = createModel([
      [
        modelOutput('reasoning', {
          finishReason: 'MAX_TOKENS',
          continuationToken: 'segment-2',
          ...usage(100, 0, 65533),
        }),
      ],
      [modelOutput('answer', { finishReason: 'STOP', ...usage(200, 5, 0) })],
    ]);

    const chunks = await collectMessages(model);

    expect(bodies).toHaveLength(1);
    expect(bodies[0].generationConfig).toMatchObject({
      maxOutputTokens: 131072,
    });
    expect(
      chunks.some(
        (chunk) => chunk.response_metadata.finish_reason === 'MAX_TOKENS'
      )
    ).toBe(true);
    const full = combineMessages(chunks);
    expect(full.response_metadata.continuation_token).toBeUndefined();
    expect(full.response_metadata.finishReason).toBe('stop');
  });

  it('does not issue another request when aborted after the first segment', async () => {
    const controller = new AbortController();
    const { model, bodies } = createModel(
      [
        [
          modelOutput('reasoning', {
            finishReason: 'CONTINUATION',
            continuationToken: 'segment-2',
            ...usage(100, 0, 65533),
          }),
        ],
        [modelOutput('answer', { finishReason: 'STOP', ...usage(200, 5, 0) })],
      ],
      {},
      true,
      (segment) => {
        if (segment === 0) controller.abort();
      }
    );

    await expect(
      collectMessages(model, { signal: controller.signal })
    ).rejects.toThrow();

    expect(bodies).toHaveLength(1);
  });

  it('continues other Gemini models when the provider returns a token', async () => {
    const { model, bodies } = createModel(
      [
        [
          modelOutput('reasoning', {
            finishReason: 'CONTINUATION',
            continuationToken: 'segment-2',
            ...usage(100, 0, 65533),
          }),
        ],
        [modelOutput('answer', { finishReason: 'STOP', ...usage(200, 5, 0) })],
      ],
      { model: 'gemini-2.5-flash' }
    );

    await collectMessages(model);

    expect(bodies).toHaveLength(2);
    expect(bodies[1].continuationToken).toBe('segment-2');
  });

  it('uses v1beta1 by default only for Gemini 4 models', () => {
    expect(apiVersion({ model: 'gemini-4-carbon' })).toBe('v1beta1');
    expect(apiVersion({ model: 'gemini-2.5-flash' })).toBe('v1');
    expect(apiVersion({ modelName: 'GEMINI-4-NEON' })).toBe('v1beta1');
    expect(apiVersion({ model: 'gemini-4-carbon', apiVersion: 'v1' })).toBe(
      'v1'
    );
  });

  it('preserves the initial prompt count after continuation without a trailing null chunk', async () => {
    const { model } = createModel(
      [
        [
          modelOutput('reasoning', {
            finishReason: 'CONTINUATION',
            continuationToken: 'segment-2',
            ...usage(100, 0, 65533),
          }),
        ],
        [modelOutput('answer', { finishReason: 'STOP', ...usage(200, 5, 0) })],
      ],
      {},
      false
    );

    const full = combineMessages(await collectMessages(model));

    expect(full.response_metadata.initial_prompt_token_count).toBe(100);
  });

  it.each([true, false])(
    'does not mark a single-segment response (trailing null chunk: %s)',
    async (withTail) => {
      const { model } = createModel(
        [
          [
            modelOutput('answer', {
              finishReason: 'STOP',
              ...usage(100, 5, 0),
            }),
          ],
        ],
        {},
        withTail
      );

      const full = combineMessages(await collectMessages(model));

      expect(full.response_metadata.initial_prompt_token_count).toBeUndefined();
    }
  );
});

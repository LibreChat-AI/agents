import { describe, it, expect, jest, afterEach } from '@jest/globals';
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type {
  GenerateContentRequest,
  GenerateContentResponse,
} from '@google/generative-ai';
import type { BaseMessage, BaseMessageChunk } from '@langchain/core/messages';
import type { LLMResult, ChatGeneration } from '@langchain/core/outputs';
import type { GoogleClientOptions, NativeMediaPort } from '@/index';
import {
  fixturePort,
  imagePart,
  imageData,
} from './__tests__/nativeMediaFixtures';
import { CustomChatGoogleGenerativeAI } from './index';
import { NativeMediaError } from './native';

type Mode = 'invoke' | 'stream' | 'streamEvents' | 'legacyEvents';
type CallOptions = Parameters<CustomChatGoogleGenerativeAI['invoke']>[1];

function fixtureHttp(
  chunks: object[][] = [[{ text: 'Hello' }]],
  overrides: Partial<GenerateContentResponse> = {}
) {
  const requests: GenerateContentRequest[] = [];
  const fetch = jest
    .spyOn(globalThis, 'fetch')
    .mockImplementation(async (input, init) => {
      const request = new Request(input, init);
      requests.push(JSON.parse(await request.text()) as GenerateContentRequest);
      const response = (parts: object[]) => ({
        candidates: [{ index: 0, content: { role: 'model', parts } }],
        usageMetadata: {
          promptTokenCount: 5,
          candidatesTokenCount: 5,
          totalTokenCount: 10,
        },
        ...overrides,
      });
      const streaming = request.url.includes(':streamGenerateContent');
      return new Response(
        streaming
          ? chunks
            .map((parts) => `data: ${JSON.stringify(response(parts))}\n\n`)
            .join('')
          : JSON.stringify(response(chunks.flat())),
        {
          headers: {
            'content-type': streaming
              ? 'text/event-stream'
              : 'application/json',
          },
        }
      );
    });
  return { requests, fetch };
}

function fixtureModel(
  port?: NativeMediaPort,
  options: Partial<GoogleClientOptions> = {}
) {
  return new CustomChatGoogleGenerativeAI({
    model: 'gemini-3-pro-image-preview',
    apiKey: 'fixture',
    nativeMedia: port,
    maxRetries: 0,
    _lc_stream_delay: 0,
    ...options,
  });
}

async function runModel(
  model: CustomChatGoogleGenerativeAI,
  mode: Mode,
  messages: BaseMessage[] = [new HumanMessage('Draw')],
  options: CallOptions = {}
): Promise<BaseMessage> {
  if (mode === 'invoke') return model.invoke(messages, options);
  if (mode === 'streamEvents') return model.streamEvents(messages, options);
  if (mode === 'legacyEvents') {
    let output: BaseMessage | undefined;
    for await (const event of model.streamEvents(messages, {
      ...options,
      version: 'v2',
    })) {
      if (event.event === 'on_chat_model_end')
        output = event.data.output as BaseMessage;
    }
    if (output == null) throw new Error('Expected model output');
    return output;
  }
  let answer: BaseMessageChunk | undefined;
  for await (const chunk of await model.stream(messages, options)) {
    answer = answer == null ? chunk : answer.concat(chunk);
  }
  if (!answer) throw new Error('Expected model output');
  return answer;
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe.each<Mode>(['invoke', 'stream', 'streamEvents', 'legacyEvents'])(
  'native media HTTP requests through %s',
  (mode) => {
    it('rejects native continuation references before HTTP when no port is configured', async () => {
      const { fetch } = fixtureHttp();
      const previous = new AIMessage({
        content: [
          {
            type: 'text',
            text: 'Saved response',
            native_media: { continuationRef: 'saved-reference' },
          },
        ],
      });
      await expect(
        runModel(fixtureModel(), mode, [previous, new HumanMessage('Continue')])
      ).rejects.toThrow('configured storage');
      expect(fetch).not.toHaveBeenCalled();
    });

    it.each([
      { responseModalities: ['TEXT'] },
      { responseModalities: ['TEXT', 'IMAGE'] },
    ])(
      'sends host-approved modalities $responseModalities on the HTTP body',
      async ({ responseModalities }) => {
        const { requests } = fixtureHttp();
        const port = fixturePort({
          start: jest.fn(async () => ({ responseModalities })),
        });
        const model = fixtureModel(port, { responseModalities: ['AUDIO'] });
        await runModel(model, mode);
        expect(requests).toHaveLength(1);
        expect(requests[0]).toMatchObject({
          generationConfig: { responseModalities },
        });
        expect(port.complete).toHaveBeenCalledTimes(1);
        expect(port.fail).not.toHaveBeenCalled();
      }
    );

    it('checks admission before HTTP and reports rejection', async () => {
      const { fetch } = fixtureHttp();
      const port = fixturePort({
        start: jest.fn(async () => {
          throw new Error('Media not permitted');
        }),
      });
      await expect(runModel(fixtureModel(port), mode)).rejects.toThrow(
        'not permitted'
      );
      expect(fetch).not.toHaveBeenCalled();
      expect(port.start).toHaveBeenCalledTimes(1);
      expect(port.part).not.toHaveBeenCalled();
      expect(port.complete).not.toHaveBeenCalled();
      expect(port.fail).toHaveBeenCalledTimes(1);
    });

    it('uses configured modalities when the host authorizes without overriding them', async () => {
      const { requests } = fixtureHttp();
      const port = fixturePort({ start: jest.fn(async () => undefined) });
      await runModel(
        fixtureModel(port, { responseModalities: ['TEXT'] }),
        mode
      );
      expect(requests[0]).toMatchObject({
        generationConfig: { responseModalities: ['TEXT'] },
      });
    });

    it('does not retain a previous invocation’s modality selection', async () => {
      const { requests } = fixtureHttp();
      const start = jest
        .fn<NativeMediaPort['start']>()
        .mockResolvedValueOnce({ responseModalities: ['TEXT', 'IMAGE'] })
        .mockResolvedValueOnce(undefined);
      const model = fixtureModel(fixturePort({ start }));
      await runModel(model, mode);
      await runModel(model, mode);
      expect(requests[0]).toMatchObject({
        generationConfig: { responseModalities: ['TEXT', 'IMAGE'] },
      });
      expect(requests[1].generationConfig).not.toHaveProperty(
        'responseModalities'
      );
    });

    it('isolates modalities and system instructions between concurrent requests', async () => {
      const { requests } = fixtureHttp();
      const start = jest
        .fn<NativeMediaPort['start']>()
        .mockResolvedValueOnce({ responseModalities: ['TEXT'] })
        .mockResolvedValueOnce({ responseModalities: ['TEXT', 'IMAGE'] });
      const model = fixtureModel(fixturePort({ start }));
      await Promise.all([
        runModel(model, mode, [
          new SystemMessage('Text only'),
          new HumanMessage('First'),
        ]),
        runModel(model, mode, [
          new SystemMessage('Images allowed'),
          new HumanMessage('Second'),
        ]),
      ]);
      expect(requests).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            generationConfig: expect.objectContaining({
              responseModalities: ['TEXT'],
            }),
            systemInstruction: {
              role: 'system',
              parts: [{ text: 'Text only' }],
            },
            contents: [{ role: 'user', parts: [{ text: 'First' }] }],
          }),
          expect.objectContaining({
            generationConfig: expect.objectContaining({
              responseModalities: ['TEXT', 'IMAGE'],
            }),
            systemInstruction: {
              role: 'system',
              parts: [{ text: 'Images allowed' }],
            },
            contents: [{ role: 'user', parts: [{ text: 'Second' }] }],
          }),
        ])
      );
    });

    it('replays signed empty text exactly after an image without mutating stored content', async () => {
      const signedEmpty = { text: '', thoughtSignature: 'signed-empty-text' };
      const { requests } = fixtureHttp([[imagePart, signedEmpty]]);
      const port = fixturePort();
      const model = fixtureModel(port);
      const initial = new HumanMessage('Draw');
      const answer = await runModel(model, mode, [initial]);
      const saved = JSON.stringify(answer);
      expect(saved).not.toContain(imageData);
      expect(saved).not.toContain('signed-empty-text');
      await runModel(model, mode, [
        initial,
        answer,
        new HumanMessage('Refine'),
      ]);
      expect(port.restore).toHaveBeenCalledTimes(2);
      expect(requests[1].contents[1]).toEqual({
        role: 'model',
        parts: [imagePart, signedEmpty],
      });
      expect(JSON.stringify(answer)).toBe(saved);
    });

    it('reports storage failure and does not mark the response complete', async () => {
      fixtureHttp([[imagePart]]);
      const port = fixturePort({
        part: jest.fn(async () => {
          throw new Error('Storage unavailable');
        }),
      });
      await expect(runModel(fixtureModel(port), mode)).rejects.toThrow(
        'Storage unavailable'
      );
      expect(port.complete).not.toHaveBeenCalled();
      expect(port.fail).toHaveBeenCalledWith(
        expect.objectContaining({ reason: 'storage' })
      );
    });

    it.each([
      { promptFeedback: { blockReason: 'SAFETY' }, candidates: [] },
      { candidates: [{ index: 0, finishReason: 'SAFETY' }] },
      { candidates: [] },
    ])(
      'rejects blocked or invalid responses before completing storage: %j',
      async (response) => {
        fixtureHttp(undefined, response as GenerateContentResponse);
        const port = fixturePort();
        await expect(runModel(fixtureModel(port), mode)).rejects.toThrow(
          'Native media provider'
        );
        expect(port.part).not.toHaveBeenCalled();
        expect(port.complete).not.toHaveBeenCalled();
        expect(port.fail).toHaveBeenCalledTimes(1);
        expect(port.fail).toHaveBeenCalledWith(
          expect.objectContaining({
            reason: 'provider',
            providerOutcome: expect.objectContaining({
              kind:
                response.candidates.length === 0 &&
                !('promptFeedback' in response)
                  ? 'invalid'
                  : 'blocked',
            }),
            usage: { input_tokens: 5, output_tokens: 5, total_tokens: 10 },
          })
        );
      }
    );

    it.each(['part', 'complete'] as const)(
      'preserves known usage once when %s persistence fails, then records retry usage independently',
      async (method) => {
        const { fetch } = fixtureHttp([[imagePart]], {
          usageMetadata: {
            promptTokenCount: 11,
            candidatesTokenCount: 1290,
            totalTokenCount: 1301,
            cachedContentTokenCount: 3,
          },
        });
        const port = fixturePort();
        jest
          .spyOn(port, method)
          .mockRejectedValueOnce(new Error('Storage unavailable'));
        const errors: Error[] = [];
        const ended = jest.fn<(output: LLMResult) => void>();
        const model = fixtureModel(port);
        const options: CallOptions = {
          callbacks: [
            {
              handleLLMError: (error: Error) => {
                errors.push(error);
              },
              handleLLMEnd: ended,
            },
          ],
        };
        await expect(runModel(model, mode, undefined, options)).rejects.toThrow(
          'Storage unavailable'
        );
        expect(ended).not.toHaveBeenCalled();
        expect(errors).toHaveLength(1);
        expect(errors[0]).toBeInstanceOf(NativeMediaError);
        const usage = {
          input_tokens: 11,
          output_tokens: 1290,
          total_tokens: 1301,
          input_token_details: { cache_read: 3 },
        };
        expect(errors[0]).toMatchObject({ usage });
        expect(port.fail).toHaveBeenCalledTimes(1);
        expect(port.fail).toHaveBeenCalledWith(
          expect.objectContaining({
            reason: 'storage',
            usage: expect.objectContaining(usage),
          })
        );
        const failure = jest.mocked(port.fail).mock.calls[0][0];
        const output = await runModel(model, mode, undefined, options);
        const completed = (
          ended.mock.calls[0][0].generations[0][0] as ChatGeneration
        ).message;
        expect(completed.additional_kwargs).toMatchObject({
          native_media_model_run_id: expect.any(String),
        });
        expect(completed.additional_kwargs.native_media_model_run_id).not.toBe(
          failure.modelRunId
        );
        expect(output).toMatchObject({ usage_metadata: usage });
        expect(ended).toHaveBeenCalledTimes(1);
        expect(port.fail).toHaveBeenCalledTimes(1);
        expect(fetch).toHaveBeenCalledTimes(2);
      }
    );

    it.each([{ text: '' }, { text: '', thoughtSignature: 'empty-signature' }])(
      'accepts valid empty text content: %j',
      async (part) => {
        fixtureHttp([[part]]);
        const port = fixturePort();
        await runModel(fixtureModel(port), mode);
        expect(port.complete).toHaveBeenCalledTimes(1);
        expect(port.fail).not.toHaveBeenCalled();
      }
    );

    it('retains failure usage when normal stream usage is disabled and failure recording also throws', async () => {
      fixtureHttp([[imagePart]]);
      const port = fixturePort({
        part: jest.fn(async () => {
          throw new Error('Original persistence error');
        }),
        fail: jest.fn(async () => {
          throw new Error('Failure recording unavailable');
        }),
      });
      await expect(
        runModel(fixtureModel(port, { streamUsage: false }), mode)
      ).rejects.toMatchObject({
        message: 'Original persistence error',
        usage: { input_tokens: 5, output_tokens: 5, total_tokens: 10 },
      });
      expect(port.fail).toHaveBeenCalledTimes(1);
    });

    it('restores batches in transcript order, detaches edited text, and preserves signed empty text', async () => {
      const { requests } = fixtureHttp();
      const port = fixturePort({
        restoreBatch: jest.fn<NonNullable<NativeMediaPort['restoreBatch']>>(
          async () => [
            {
              kind: 'text',
              text: 'Original',
              thoughtSignature: 'original-signature',
            },
            {
              kind: 'image',
              ...imagePart.inlineData,
              thoughtSignature: imagePart.thoughtSignature,
            },
            { kind: 'text', text: '', thoughtSignature: 'empty-signature' },
          ]
        ),
      });
      const previous = new AIMessage({
        content: [
          {
            type: 'text',
            text: 'Corrected',
            native_media: { continuationRef: 'text' },
          },
          {
            type: 'image_file',
            image_file: { file_id: 'image-file' },
            native_media: { continuationRef: 'image' },
          },
          {
            type: 'text',
            text: '',
            native_media: { continuationRef: 'empty' },
          },
        ],
      });
      const saved = JSON.stringify(previous);
      await runModel(fixtureModel(port), mode, [
        new HumanMessage('Draw'),
        previous,
        new HumanMessage('Continue'),
      ]);
      expect(port.restoreBatch).toHaveBeenCalledTimes(1);
      expect(port.restoreBatch).toHaveBeenCalledWith(
        expect.objectContaining({
          parts: [
            { continuationRef: 'text' },
            { file_id: 'image-file', continuationRef: 'image' },
            { continuationRef: 'empty' },
          ],
        })
      );
      expect(port.restore).not.toHaveBeenCalled();
      expect(requests[0].contents[1].parts).toEqual([
        { text: 'Corrected' },
        imagePart,
        { text: '', thoughtSignature: 'empty-signature' },
      ]);
      expect(JSON.stringify(previous)).toBe(saved);
    });

    it.each(['incomplete', 'incompatible', 'unauthorized'] as const)(
      'rejects %s batches before provider HTTP',
      async (failure) => {
        const { fetch } = fixtureHttp();
        const port = fixturePort({
          restoreBatch: jest.fn<NonNullable<NativeMediaPort['restoreBatch']>>(
            async () => {
              if (failure === 'unauthorized')
                throw new Error('Unauthorized continuation');
              return failure === 'incomplete'
                ? []
                : [{ kind: 'image', ...imagePart.inlineData }];
            }
          ),
        });
        await expect(
          runModel(fixtureModel(port), mode, [
            new AIMessage({
              content: [
                {
                  type: 'text',
                  text: 'Previous',
                  native_media: { continuationRef: 'text' },
                },
              ],
            }),
            new HumanMessage('Continue'),
          ])
        ).rejects.toThrow();
        expect(fetch).not.toHaveBeenCalled();
        expect(port.complete).not.toHaveBeenCalled();
      }
    );

    it('reports cancellation while storage is pending', async () => {
      fixtureHttp([[imagePart]]);
      const controller = new AbortController();
      const storage = fixturePort();
      const port = fixturePort({
        part: jest.fn(async (input: Parameters<NativeMediaPort['part']>[0]) => {
          const result = await storage.part(input);
          controller.abort();
          return result;
        }),
      });
      await expect(
        runModel(fixtureModel(port), mode, undefined, {
          signal: controller.signal,
        })
      ).rejects.toThrow();
      expect(port.complete).not.toHaveBeenCalled();
      expect(port.fail).toHaveBeenCalledWith(
        expect.objectContaining({
          reason: 'aborted',
          usage: { input_tokens: 5, output_tokens: 5, total_tokens: 10 },
        })
      );
    });
  }
);

it('streams durable mixed media through the public event API before assembling replayable output', async () => {
  fixtureHttp([
    [{ text: 'Before', thoughtSignature: 'signed-text' }, imagePart],
    [{ text: 'After' }],
  ]);
  const port = fixturePort();
  const stream = fixtureModel(port).streamEvents('Draw');
  const emitted = [];
  for await (const event of stream) {
    if (event.event === 'content-block-start') {
      expect(port.part).toHaveBeenCalled();
      emitted.push(event.content);
    }
  }
  const output = await stream;
  expect(emitted.map((part) => part.type)).toEqual([
    'text',
    'image_file',
    'text',
  ]);
  expect(output.content).toEqual(emitted);
  expect(output.usage_metadata).toMatchObject({
    input_tokens: 5,
    output_tokens: 5,
    total_tokens: 10,
  });
  expect(JSON.stringify(output)).not.toContain(imageData);
  expect(JSON.stringify(output)).not.toContain('private-signature');
  expect(JSON.stringify(output)).not.toContain('signed-text');
  expect(port.part).toHaveBeenCalledTimes(3);
  expect(port.complete).toHaveBeenCalledTimes(1);
});

it('keeps unsigned text around images in order when the host omits text replay markers', async () => {
  fixtureHttp([[{ text: 'Before' }], [imagePart], [{ text: 'After' }]]);
  const storage = fixturePort();
  const port = fixturePort({
    part: jest.fn(async (input: Parameters<NativeMediaPort['part']>[0]) => {
      if (input.part.kind === 'text')
        return { type: 'text' as const, text: input.part.text };
      return storage.part(input);
    }),
  });
  const output = await fixtureModel(port).streamEvents('Draw');
  expect(output.content).toEqual([
    { type: 'text', text: 'Before' },
    expect.objectContaining({ type: 'image_file' }),
    { type: 'text', text: 'After' },
  ]);
});

it.each<Mode>(['stream', 'streamEvents'])(
  'preserves adjacent signed text and continuation references under default smoothing in %s',
  async (mode) => {
    const parts = [
      {
        text: 'The first long piece of signed text comes before the image.',
        thoughtSignature: 'first-signature',
      },
      {
        text: 'A second long piece of signed text follows the first.',
        thoughtSignature: 'second-signature',
      },
      imagePart,
    ];
    const { requests } = fixtureHttp(parts.map((part) => [part]));
    const model = fixtureModel(fixturePort(), { _lc_stream_delay: undefined });
    const initial = new HumanMessage('Draw');
    const answer = await runModel(model, mode, [initial]);
    expect(Array.isArray(answer.content) && answer.content).toHaveLength(3);
    await runModel(model, mode, [initial, answer, new HumanMessage('Refine')]);
    expect(requests[1].contents[1]).toEqual({ role: 'model', parts });
  }
);

it.each<Mode>(['invoke', 'stream'])(
  'preserves existing server-tool signatures without a port in %s',
  async (mode) => {
    const parts = [
      {
        toolCall: { id: 'search-1', name: 'google_search', args: {} },
        thoughtSignature: 'tool-signature',
      },
      {
        toolResponse: {
          id: 'search-1',
          name: 'google_search',
          response: { results: [] },
        },
        thoughtSignature: 'result-signature',
      },
      { text: 'Search completed' },
    ];
    const { requests } = fixtureHttp([parts]);
    const model = fixtureModel(undefined, {
      model: 'gemini-3-flash-preview',
      includeServerSideToolInvocations: true,
    });
    const initial = new HumanMessage('Search');
    const answer = await runModel(model, mode, [initial]);
    await runModel(model, mode, [
      initial,
      answer,
      new HumanMessage('Continue'),
    ]);
    expect(requests[1].contents[1]).toEqual({ role: 'model', parts });
  }
);

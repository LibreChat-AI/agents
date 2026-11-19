import { createParser } from 'eventsource-parser';
import { GoogleGenerativeAIFetchError } from '@google/generative-ai';
import type {
  Content,
  Part,
  GenerateContentResult,
  ErrorDetails,
  GenerateContentRequest,
  GenerateContentResponse,
  ModelParams,
  RequestOptions,
  SingleRequestOptions,
} from '@google/generative-ai';
import type { EventSourceParser } from 'eventsource-parser';
import { enhanceResponse, formatRequest } from './completion';

type GoogleError = {
  code?: number;
  message?: string;
  status?: string;
  details?: ErrorDetails[];
};

type GoogleResponse = GenerateContentResponse & { error?: GoogleError };

function providerError(
  error?: GoogleError,
  status = 500,
  statusText = ''
): GoogleGenerativeAIFetchError {
  return new GoogleGenerativeAIFetchError(
    error?.message ?? `Google request failed: ${status} ${statusText}`,
    status,
    statusText,
    error?.details
  );
}

function formatSystemInstruction(
  input: GenerateContentRequest['systemInstruction']
): Content | undefined {
  if (input == null) return undefined;
  if (typeof input === 'string') {
    return { role: 'system', parts: [{ text: input }] };
  }
  if ('text' in input && input.text != null && input.text !== '') {
    return { role: 'system', parts: [input] };
  }
  if ('parts' in input) {
    return { ...input, role: input.role || 'system' };
  }
  return undefined;
}

/** Follows LangChainJS's GenerateContentApiHandler fetch/SSE flow, retaining our wire contract. */
export class GoogleTransport {
  readonly generationConfig: NonNullable<ModelParams['generationConfig']>;

  constructor(
    private readonly apiKey: string,
    private readonly params: ModelParams,
    private readonly options: RequestOptions = {},
    private readonly cachedContent?: string
  ) {
    this.generationConfig = params.generationConfig ?? {};
  }

  private async request(
    request: GenerateContentRequest,
    streaming: boolean,
    options: SingleRequestOptions = {}
  ): Promise<Response> {
    const { model, cachedContent, ...defaults } = this.params;
    const cacheName = this.cachedContent ?? cachedContent?.name;
    const requestOptions = { ...this.options, ...options };
    const baseUrl = (
      requestOptions.baseUrl ?? 'https://generativelanguage.googleapis.com'
    ).replace(/\/$/, '');
    const apiVersion = requestOptions.apiVersion ?? 'v1beta';
    const method = streaming
      ? 'streamGenerateContent?alt=sse'
      : 'generateContent';
    const url = `${baseUrl}/${apiVersion}/models/${model.replace(/^models\//, '')}:${method}`;
    const headers = new Headers({
      'content-type': 'application/json',
      'x-goog-api-key': this.apiKey,
    });
    if (requestOptions.apiClient != null) {
      headers.set('x-goog-api-client', requestOptions.apiClient);
    }
    new Headers(requestOptions.customHeaders).forEach((value, name) => {
      headers.set(name, value);
    });
    const timeout = requestOptions.timeout;
    const timeoutSignal =
      timeout != null && timeout >= 0
        ? AbortSignal.timeout(Math.trunc(timeout))
        : undefined;
    const signal =
      options.signal && timeoutSignal
        ? AbortSignal.any([options.signal, timeoutSignal])
        : (options.signal ?? timeoutSignal);
    const body: GenerateContentRequest = {
      ...defaults,
      ...(cacheName != null && cacheName !== ''
        ? { cachedContent: cacheName }
        : {}),
      ...request,
      systemInstruction: formatSystemInstruction(
        request.systemInstruction ?? defaults.systemInstruction
      ),
    };
    const response = await fetch(
      new Request(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal,
      })
    );
    if (!response.ok) {
      const payload: GoogleResponse | undefined = await response
        .json()
        .catch(() => undefined);
      throw providerError(payload?.error, response.status, response.statusText);
    }
    return response;
  }

  async generateContent(
    request: GenerateContentRequest | string | (string | Part)[],
    options?: SingleRequestOptions
  ): Promise<GenerateContentResult> {
    const response = await this.request(formatRequest(request), false, options);
    const payload: GoogleResponse = await response.json();
    if (payload.error) throw providerError(payload.error, payload.error.code);
    return { response: enhanceResponse(payload) };
  }

  async generateContentStream(
    request: GenerateContentRequest,
    options?: SingleRequestOptions
  ): Promise<{ stream: AsyncGenerator<GenerateContentResponse> }> {
    const controller = new AbortController();
    const signal = options?.signal
      ? AbortSignal.any([options.signal, controller.signal])
      : controller.signal;
    try {
      const response = await this.request(request, true, {
        ...options,
        signal,
      });
      if (response.body == null)
        throw new Error('Google response body is empty');
      const decoder = new TextDecoder('utf-8', { fatal: true });
      let parser: EventSourceParser;
      const reader = response.body
        .pipeThrough(
          new TransformStream<Uint8Array, GenerateContentResponse>({
            start(output): void {
              parser = createParser({
                onEvent(event): void {
                  const payload: GoogleResponse = JSON.parse(event.data);
                  if (payload.error)
                    throw providerError(payload.error, payload.error.code);
                  output.enqueue(payload);
                },
                onError(error): void {
                  throw error;
                },
              });
            },
            transform(chunk): void {
              parser.feed(decoder.decode(chunk, { stream: true }));
            },
            flush(): void {
              parser.feed(decoder.decode());
              parser.feed('\n\n');
            },
          })
        )
        .getReader();
      async function* stream(): AsyncGenerator<GenerateContentResponse> {
        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) return;
            yield value;
          }
        } finally {
          controller.abort();
          await reader.cancel().catch(() => undefined);
          reader.releaseLock();
        }
      }
      return { stream: stream() };
    } catch (error) {
      controller.abort();
      throw error;
    }
  }
}

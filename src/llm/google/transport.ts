import { GoogleGenAI } from '@google/genai';
import type {
  GenerateContentRequest,
  GenerateContentResponse,
  ModelParams,
  RequestOptions,
  SingleRequestOptions,
} from '@google/generative-ai';
import type { GenerateContentParameters } from '@google/genai';

/** Retains LangChain's wire contract without using its archived SDK transport. */
export class GoogleTransport {
  readonly generationConfig: NonNullable<ModelParams['generationConfig']>;
  private readonly client: GoogleGenAI;

  constructor(
    apiKey: string,
    private readonly params: ModelParams,
    private readonly options: RequestOptions = {},
    private readonly cachedContent?: string
  ) {
    this.generationConfig = params.generationConfig ?? {};
    this.client = new GoogleGenAI({
      apiKey,
      vertexai: false,
      httpOptions: {
        baseUrl: options.baseUrl ?? 'https://generativelanguage.googleapis.com',
        apiVersion: options.apiVersion ?? 'v1beta',
      },
    });
  }

  private request(
    request: GenerateContentRequest,
    options: SingleRequestOptions = {}
  ): GenerateContentParameters {
    const { model, ...defaults } = this.params;
    const requestOptions = { ...this.options, ...options };
    return {
      model,
      contents: [{ role: 'user', parts: [] }],
      config: {
        abortSignal: options.signal,
        automaticContinuation: false,
        automaticFunctionCalling: { disable: true },
        httpOptions: {
          baseUrl: (
            requestOptions.baseUrl ??
            'https://generativelanguage.googleapis.com'
          ).replace(/\/$/, ''),
          apiVersion: requestOptions.apiVersion ?? 'v1beta',
          headers: Object.fromEntries(
            new Headers(requestOptions.customHeaders)
          ),
          timeout: requestOptions.timeout,
          // Raw fields bypass SDK conversion; this also replaces the placeholder contents.
          extraBody: {
            ...defaults,
            ...request,
            ...(this.cachedContent != null && this.cachedContent !== ''
              ? { cachedContent: this.cachedContent }
              : {}),
          },
        },
      },
    };
  }

  async generateContent(
    request: GenerateContentRequest,
    options?: SingleRequestOptions
  ): Promise<{ response: GenerateContentResponse }> {
    const response = await this.client.models.generateContent(
      this.request(request, options)
    );
    return { response: response as GenerateContentResponse };
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
      const source = await this.client.models.generateContentStream(
        this.request(request, { ...options, signal })
      );
      async function* stream(): AsyncGenerator<GenerateContentResponse> {
        try {
          for await (const response of source) {
            yield response as GenerateContentResponse;
          }
        } finally {
          controller.abort();
        }
      }
      return { stream: stream() };
    } catch (error) {
      controller.abort();
      throw error;
    }
  }
}

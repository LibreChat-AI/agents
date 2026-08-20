const mockLangfuseSpanProcessorInstance = {};
type MockLangfuseSpanProcessorParams = {
  mask?: (input: { data: unknown }) => string;
};
const mockLangfuseSpanProcessor = jest.fn(
  (_params?: MockLangfuseSpanProcessorParams) =>
    mockLangfuseSpanProcessorInstance
);
const mockSetLangfuseTracerProvider = jest.fn();
let mockContextHasActiveValue = false;
const mockContextKey = Symbol('mock-context-key');
const mockContextActive = jest.fn(() => ({
  getValue: jest.fn(() => mockContextHasActiveValue),
}));
const mockContextWith = jest.fn(
  (_contextValue: unknown, callback: () => boolean) => callback()
);
const mockSetGlobalContextManager = jest.fn(() => true);
const mockRootContext = {
  setValue: jest.fn(() => ({})),
};
const mockContextManager = {
  disable: jest.fn(),
  enable: jest.fn(),
};
const mockAsyncLocalStorageContextManager = jest.fn(() => mockContextManager);
const mockTracerProvider = {
  forceFlush: jest.fn(),
  getTracer: jest.fn(),
  shutdown: jest.fn(),
};
type BasicTracerProviderInput = {
  spanProcessors: Array<{
    forceFlush?: unknown;
    onEnd?: unknown;
    onStart?: unknown;
    shutdown?: unknown;
  }>;
};
type RoutingSpanProcessorForTest =
  BasicTracerProviderInput['spanProcessors'][0] & {
    processors: Map<
      string,
      {
        fallbackConfig?: {
          enabled?: boolean;
        };
      }
    >;
  };
const mockBasicTracerProvider = jest.fn(
  (_input?: BasicTracerProviderInput) => mockTracerProvider
);

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: mockLangfuseSpanProcessor,
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@langfuse/tracing', () => ({
  ...jest.requireActual('@langfuse/tracing'),
  setLangfuseTracerProvider: mockSetLangfuseTracerProvider,
}));

jest.mock('@opentelemetry/api', () => ({
  ...jest.requireActual('@opentelemetry/api'),
  context: {
    ...jest.requireActual('@opentelemetry/api').context,
    active: mockContextActive,
    setGlobalContextManager: mockSetGlobalContextManager,
    with: mockContextWith,
  },
  createContextKey: jest.fn(() => mockContextKey),
  ROOT_CONTEXT: mockRootContext,
}));

jest.mock('@opentelemetry/context-async-hooks', () => ({
  AsyncLocalStorageContextManager: mockAsyncLocalStorageContextManager,
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: mockBasicTracerProvider,
  SpanStatusCode: jest.requireActual('@opentelemetry/api').SpanStatusCode,
}));

describe('Langfuse instrumentation', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    mockContextHasActiveValue = false;
    mockSetGlobalContextManager.mockReturnValue(true);
    process.env = { ...originalEnv };
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('does not initialize tracing when Langfuse env vars are missing', async () => {
    const { initializeLangfuseTracingFromEnv } = await import(
      '@/instrumentation'
    );

    expect(initializeLangfuseTracingFromEnv()).toBeUndefined();
    expect(mockLangfuseSpanProcessor).not.toHaveBeenCalled();
    expect(mockBasicTracerProvider).not.toHaveBeenCalled();
    expect(mockAsyncLocalStorageContextManager).not.toHaveBeenCalled();
    expect(mockSetLangfuseTracerProvider).not.toHaveBeenCalled();
  });

  it('registers an isolated Langfuse tracer provider from env config', async () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    const { initializeLangfuseTracingFromEnv } = await import(
      '@/instrumentation'
    );
    const provider = initializeLangfuseTracingFromEnv();

    expect(provider).toBe(mockTracerProvider);
    expect(mockLangfuseSpanProcessor).toHaveBeenCalledTimes(1);
    const providerInput = mockBasicTracerProvider.mock
      .calls[0][0] as BasicTracerProviderInput;
    expect(providerInput.spanProcessors).toHaveLength(1);
    expect(providerInput.spanProcessors[0]).not.toBe(
      mockLangfuseSpanProcessorInstance
    );
    expect(providerInput.spanProcessors[0]).toMatchObject({
      forceFlush: expect.any(Function),
      onEnd: expect.any(Function),
      onStart: expect.any(Function),
      shutdown: expect.any(Function),
    });
    expect(mockSetLangfuseTracerProvider).toHaveBeenCalledWith(
      mockTracerProvider
    );
    expect(mockAsyncLocalStorageContextManager).toHaveBeenCalledTimes(1);
    expect(mockContextManager.enable).toHaveBeenCalledTimes(1);
    expect(mockSetGlobalContextManager).toHaveBeenCalledWith(
      mockContextManager
    );
  });

  it('registers tracing from explicit Langfuse config credentials', async () => {
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const provider = initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
    });

    expect(provider).toBe(mockTracerProvider);
    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        publicKey: 'pk-config',
        secretKey: 'sk-config',
        baseUrl: 'https://langfuse.config',
      })
    );
    expect(mockBasicTracerProvider).toHaveBeenCalledTimes(1);
  });

  it('uses env credentials with a config-provided Langfuse baseUrl', async () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';

    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const provider = initializeLangfuseTracing({
      baseUrl: 'https://langfuse.config',
      toolOutputTracing: { enabled: false },
    });

    expect(provider).toBe(mockTracerProvider);
    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        publicKey: 'pk-env',
        secretKey: 'sk-env',
        baseUrl: 'https://langfuse.config',
      })
    );
    expect(mockBasicTracerProvider).toHaveBeenCalledTimes(1);
  });

  it('resolves environment from LANGFUSE_TRACING_ENVIRONMENT', async () => {
    process.env.LANGFUSE_TRACING_ENVIRONMENT = 'staging';

    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({ environment: 'staging' })
    );
  });

  it('constructs the span processor with a privacy mask under metricsOnly', async () => {
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
      privacy: { mode: 'metricsOnly', redactionText: '[private]' },
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        mediaUploadEnabled: false,
        mask: expect.any(Function),
      })
    );
    const params = mockLangfuseSpanProcessor.mock.calls[0][0];
    expect(params?.mask?.({ data: 'conversation text' })).toBe('[private]');
  });

  it('falls back to NODE_ENV when LANGFUSE_TRACING_ENVIRONMENT is unset', async () => {
    delete process.env.LANGFUSE_TRACING_ENVIRONMENT;
    process.env.NODE_ENV = 'production';

    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({ environment: 'production' })
    );
  });

  it('falls through blank environment overrides and trims the selected value', async () => {
    process.env.LANGFUSE_TRACING_ENVIRONMENT = '   ';
    process.env.NODE_ENV = ' production ';

    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
      environment: '',
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({ environment: 'production' })
    );
  });

  it('prefers an explicit config environment over environment variables', async () => {
    process.env.LANGFUSE_TRACING_ENVIRONMENT = 'staging';

    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
      environment: 'canary',
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({ environment: 'canary' })
    );
  });

  it('does not replace the global provider when explicit credentials change', async () => {
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-first',
      secretKey: 'sk-first',
      baseUrl: 'https://langfuse.first',
    });
    initializeLangfuseTracing({
      publicKey: 'pk-second',
      secretKey: 'sk-second',
      baseUrl: 'https://langfuse.second',
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledTimes(2);
    expect(mockLangfuseSpanProcessor).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        publicKey: 'pk-first',
        secretKey: 'sk-first',
        baseUrl: 'https://langfuse.first',
      })
    );
    expect(mockLangfuseSpanProcessor).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        publicKey: 'pk-second',
        secretKey: 'sk-second',
        baseUrl: 'https://langfuse.second',
      })
    );
    expect(mockBasicTracerProvider).toHaveBeenCalledTimes(1);
    expect(mockSetLangfuseTracerProvider).toHaveBeenCalledTimes(1);
  });

  it('does not store raw secret keys in processor cache keys', async () => {
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-cache',
      secretKey: 'sk-cache-secret',
      baseUrl: 'https://langfuse.cache',
    });

    const providerInput = mockBasicTracerProvider.mock
      .calls[0][0] as BasicTracerProviderInput;
    const routingProcessor = providerInput
      .spanProcessors[0] as RoutingSpanProcessorForTest;
    const processorKeys = Array.from(routingProcessor.processors.keys());

    expect(processorKeys).toHaveLength(1);
    expect(processorKeys[0]).toContain('secretKeyHash');
    expect(processorKeys[0]).not.toContain('sk-cache-secret');
  });

  it('passes explicit redaction config into the redacting processor fallback', async () => {
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-config',
      secretKey: 'sk-config',
      baseUrl: 'https://langfuse.config',
      toolOutputTracing: { enabled: false },
    });

    const providerInput = mockBasicTracerProvider.mock
      .calls[0][0] as BasicTracerProviderInput;
    const routingProcessor = providerInput
      .spanProcessors[0] as RoutingSpanProcessorForTest;
    const childProcessors = Array.from(routingProcessor.processors.values());
    expect(childProcessors[0]?.fallbackConfig).toMatchObject({
      enabled: false,
    });
  });

  it('does not reuse processors across different media upload policies', async () => {
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-media',
      secretKey: 'sk-media',
      baseUrl: 'https://langfuse.media',
      mediaUploadEnabled: false,
    });
    initializeLangfuseTracing({
      publicKey: 'pk-media',
      secretKey: 'sk-media',
      baseUrl: 'https://langfuse.media',
      mediaUploadEnabled: true,
    });

    expect(mockLangfuseSpanProcessor).toHaveBeenCalledTimes(2);
    expect(mockLangfuseSpanProcessor).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ mediaUploadEnabled: false })
    );
    expect(mockLangfuseSpanProcessor).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ mediaUploadEnabled: true })
    );
  });

  it('reuses the isolated provider after initialization', async () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    const { initializeLangfuseTracingFromEnv } = await import(
      '@/instrumentation'
    );
    const firstProvider = initializeLangfuseTracingFromEnv();
    const secondProvider = initializeLangfuseTracingFromEnv();

    expect(firstProvider).toBe(mockTracerProvider);
    expect(secondProvider).toBe(mockTracerProvider);
    expect(mockLangfuseSpanProcessor).toHaveBeenCalledTimes(1);
    expect(mockBasicTracerProvider).toHaveBeenCalledTimes(1);
    expect(mockSetLangfuseTracerProvider).toHaveBeenCalledTimes(1);
  });

  it('does not replace an existing active context manager', async () => {
    mockContextHasActiveValue = true;
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    await import('@/instrumentation');

    expect(mockAsyncLocalStorageContextManager).not.toHaveBeenCalled();
    expect(mockSetGlobalContextManager).not.toHaveBeenCalled();
  });

  it('disables the local context manager when registration is rejected', async () => {
    mockSetGlobalContextManager.mockReturnValue(false);
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    await import('@/instrumentation');

    expect(mockContextManager.enable).toHaveBeenCalledTimes(1);
    expect(mockSetGlobalContextManager).toHaveBeenCalledWith(
      mockContextManager
    );
    expect(mockContextManager.disable).toHaveBeenCalledTimes(1);
  });
});

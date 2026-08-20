import type { Span } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  getLangfuseSpanProcessorParams,
  getLangfuseManagedSpanDestination,
  registerLangfuseTraceAnchorSpan,
  resolveLangfuseTraceAnchorParent,
  registerLangfuseManagedSpan,
  resolveLangfuseDestinationKey,
} from '@/langfuseSpanRegistry';

const ORIGINAL_ENV = { ...process.env };

function tenantConfig(
  overrides: Partial<t.LangfuseConfig> = {}
): t.LangfuseConfig {
  return {
    enabled: true,
    publicKey: 'pk-registry',
    secretKey: 'sk-registry',
    baseUrl: 'https://langfuse.registry',
    ...overrides,
  };
}

describe('Langfuse span registry', () => {
  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  it('tracks managed spans by destination', () => {
    const span = { name: 'managed' } as unknown as Span;
    const key = resolveLangfuseDestinationKey(tenantConfig());
    expect(key).toBeDefined();
    registerLangfuseManagedSpan(span, key as string);
    expect(getLangfuseManagedSpanDestination(span)).toBe(key);

    const untracked = { name: 'foreign' } as unknown as Span;
    expect(getLangfuseManagedSpanDestination(untracked)).toBeUndefined();
  });

  it('resolves run anchors only for their export destination', () => {
    const anchor = {};
    const spanContext = {
      traceId: '0123456789abcdef0123456789abcdef',
      spanId: '0123456789abcdef',
      traceFlags: 1,
    };
    const span = {
      spanContext: () => spanContext,
    } as unknown as Span;
    const destinationKey = resolveLangfuseDestinationKey(tenantConfig());

    registerLangfuseTraceAnchorSpan(anchor, span, destinationKey as string);

    expect(resolveLangfuseTraceAnchorParent(anchor, destinationKey)).toEqual(
      spanContext
    );
    expect(
      resolveLangfuseTraceAnchorParent(anchor, 'another-destination')
    ).toBeUndefined();
  });

  it('treats processor policy as irrelevant to destination identity', () => {
    const base = resolveLangfuseDestinationKey(tenantConfig());
    const redacting = resolveLangfuseDestinationKey(
      tenantConfig({ toolOutputTracing: { enabled: false } })
    );
    const mediaDisabled = resolveLangfuseDestinationKey(
      tenantConfig({ mediaUploadEnabled: false })
    );
    const privacyOnly = resolveLangfuseDestinationKey(
      tenantConfig({ privacy: { mode: 'metricsOnly' } })
    );
    expect(redacting).toBe(base);
    expect(mediaDisabled).toBe(base);
    expect(privacyOnly).toBe(base);
  });

  it('passes media upload policy to the Langfuse span processor params', () => {
    expect(
      getLangfuseSpanProcessorParams(
        tenantConfig({ mediaUploadEnabled: false })
      )
    ).toEqual(
      expect.objectContaining({
        mediaUploadEnabled: false,
      })
    );
  });

  it('masks content and disables media upload under metricsOnly privacy', () => {
    const params = getLangfuseSpanProcessorParams(
      tenantConfig({
        mediaUploadEnabled: true,
        privacy: { mode: 'metricsOnly' },
      })
    );
    expect(params).toEqual(
      expect.objectContaining({
        mediaUploadEnabled: false,
      })
    );
    expect(typeof params?.mask).toBe('function');
    expect(params?.mask?.({ data: { messages: ['secret'] } })).toBe(
      '[CONTENT REDACTED]'
    );
  });

  it('uses the configured redaction text in the privacy mask', () => {
    const params = getLangfuseSpanProcessorParams(
      tenantConfig({
        privacy: { mode: 'metricsOnly', redactionText: '[private]' },
      })
    );
    expect(params?.mask?.({ data: 'prompt text' })).toBe('[private]');
  });

  it('leaves content params untouched in full privacy mode', () => {
    const params = getLangfuseSpanProcessorParams(
      tenantConfig({ mediaUploadEnabled: true, privacy: { mode: 'full' } })
    );
    expect(params?.mediaUploadEnabled).toBe(true);
    expect(params?.mask).toBeUndefined();
  });

  it('separates destinations by credentials, endpoint, and environment', () => {
    const base = resolveLangfuseDestinationKey(tenantConfig());
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ publicKey: 'pk-other' }))
    ).not.toBe(base);
    expect(
      resolveLangfuseDestinationKey(
        tenantConfig({ baseUrl: 'https://langfuse.other' })
      )
    ).not.toBe(base);
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ environment: 'staging' }))
    ).not.toBe(base);
  });

  it('resolves no destination without credentials', () => {
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    expect(resolveLangfuseDestinationKey(undefined)).toBeUndefined();
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ enabled: false }))
    ).toBeUndefined();
  });

  it('passes custom headers to the Langfuse span processor params', () => {
    expect(
      getLangfuseSpanProcessorParams(
        tenantConfig({ additionalHeaders: { 'CF-Access-Client-Id': 'proxy' } })
      )
    ).toEqual(
      expect.objectContaining({
        additionalHeaders: { 'CF-Access-Client-Id': 'proxy' },
      })
    );
  });

  it('passes custom headers alongside env credentials', () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;

    expect(
      getLangfuseSpanProcessorParams({
        additionalHeaders: { 'X-Proxy-Token': 'env-branch' },
      })
    ).toEqual(
      expect.objectContaining({
        publicKey: 'pk-env',
        secretKey: 'sk-env',
        additionalHeaders: { 'X-Proxy-Token': 'env-branch' },
      })
    );

    expect(
      getLangfuseSpanProcessorParams({
        baseUrl: 'https://langfuse.self-hosted',
        additionalHeaders: { 'X-Proxy-Token': 'baseurl-branch' },
      })
    ).toEqual(
      expect.objectContaining({
        baseUrl: 'https://langfuse.self-hosted',
        additionalHeaders: { 'X-Proxy-Token': 'baseurl-branch' },
      })
    );
  });

  it('omits custom headers from the params when none are configured', () => {
    expect(getLangfuseSpanProcessorParams(tenantConfig())).not.toHaveProperty(
      'additionalHeaders'
    );
    expect(
      getLangfuseSpanProcessorParams(tenantConfig({ additionalHeaders: {} }))
    ).not.toHaveProperty('additionalHeaders');
  });

  it('separates destinations by custom headers', () => {
    const base = resolveLangfuseDestinationKey(tenantConfig());
    const proxied = resolveLangfuseDestinationKey(
      tenantConfig({ additionalHeaders: { 'X-Proxy-Token': 'first' } })
    );
    expect(proxied).not.toBe(base);
    expect(
      resolveLangfuseDestinationKey(
        tenantConfig({ additionalHeaders: { 'X-Proxy-Token': 'rotated' } })
      )
    ).not.toBe(proxied);
    expect(
      resolveLangfuseDestinationKey(
        tenantConfig({ additionalHeaders: { 'X-Tenant': 'first' } })
      )
    ).not.toBe(proxied);
  });

  it('keeps one destination across header key order, casing, and empty maps', () => {
    const base = resolveLangfuseDestinationKey(tenantConfig());
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ additionalHeaders: {} }))
    ).toBe(base);

    const ordered = resolveLangfuseDestinationKey(
      tenantConfig({
        additionalHeaders: { 'X-Alpha': 'a', 'X-Beta': 'b' },
      })
    );
    expect(
      resolveLangfuseDestinationKey(
        tenantConfig({
          additionalHeaders: { 'X-Beta': 'b', 'X-Alpha': 'a' },
        })
      )
    ).toBe(ordered);
    expect(
      resolveLangfuseDestinationKey(
        tenantConfig({
          additionalHeaders: { 'x-alpha': 'a', 'x-beta': 'b' },
        })
      )
    ).toBe(ordered);
  });

  it('keeps header values out of the destination key', () => {
    const key = resolveLangfuseDestinationKey(
      tenantConfig({ additionalHeaders: { 'X-Proxy-Token': 'super-secret' } })
    );
    expect(key).toBeDefined();
    expect(key).not.toContain('super-secret');
  });
});

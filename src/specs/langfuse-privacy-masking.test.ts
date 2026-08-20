import { LangfuseSpanProcessor } from '@langfuse/otel';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import type {
  ReadableSpan,
  Span,
  SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import type { ExportResult } from '@opentelemetry/core';
import { getLangfuseSpanProcessorParams } from '@/langfuseSpanRegistry';

class InMemoryExporter implements SpanExporter {
  readonly exportedSpans: ReadableSpan[] = [];

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this.exportedSpans.push(...spans);
    resultCallback({ code: 0 });
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }
}

/**
 * Drives the real `LangfuseSpanProcessor` (no `@langfuse/otel` mocks) with the
 * params `getLangfuseSpanProcessorParams` builds under `metricsOnly`, so the
 * test proves the registry's privacy wiring reaches the SDK's masking layer
 * and that operational attributes survive it.
 */
describe('Langfuse privacy masking', () => {
  async function exportSpanWithPrivacy(
    privacy: { mode: 'full' | 'metricsOnly'; redactionText?: string },
    attributes: Record<string, unknown>
  ): Promise<ReadableSpan | undefined> {
    const exporter = new InMemoryExporter();
    const params = getLangfuseSpanProcessorParams({
      publicKey: 'pk-privacy',
      secretKey: 'sk-privacy',
      baseUrl: 'https://langfuse.privacy',
      privacy,
    });
    if (params == null) {
      throw new Error(
        'expected span processor params for configured credentials'
      );
    }
    const processor = new LangfuseSpanProcessor({
      ...params,
      exporter,
      shouldExportSpan: () => true,
    });
    const provider = new BasicTracerProvider({ spanProcessors: [processor] });

    const span = provider
      .getTracer('privacy-masking-test')
      .startSpan('ChatGeneration') as Span;
    span.setAttributes(attributes);
    span.end();
    await processor.forceFlush();
    await provider.shutdown();
    return exporter.exportedSpans[0];
  }

  it('replaces content attributes while keeping operational data', async () => {
    const span = await exportSpanWithPrivacy(
      { mode: 'metricsOnly', redactionText: '[private]' },
      {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
          { role: 'user', content: 'What is my API key sk-secret?' },
        ]),
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'Here is your key',
        [LangfuseOtelSpanAttributes.OBSERVATION_METADATA]: JSON.stringify({
          tenantId: 'tenant-7',
        }),
        [LangfuseOtelSpanAttributes.TRACE_METADATA]: JSON.stringify({
          source: 'api',
        }),
        [LangfuseOtelSpanAttributes.OBSERVATION_MODEL]: 'gpt-5.2',
        [LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS]: JSON.stringify({
          input: 120,
          output: 34,
        }),
      }
    );

    expect(span).toBeDefined();
    expect(span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]).toBe(
      '[private]'
    );
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
    ).toBe('[private]');
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_METADATA]
    ).toBe('[private]');
    expect(span?.attributes[LangfuseOtelSpanAttributes.TRACE_METADATA]).toBe(
      '[private]'
    );
    expect(span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL]).toBe(
      'gpt-5.2'
    );
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS]
    ).toBe(JSON.stringify({ input: 120, output: 34 }));
  });

  it('exports content unchanged in full mode', async () => {
    const span = await exportSpanWithPrivacy(
      { mode: 'full' },
      {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: 'plain prompt',
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'plain completion',
      }
    );

    expect(span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]).toBe(
      'plain prompt'
    );
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
    ).toBe('plain completion');
  });
});

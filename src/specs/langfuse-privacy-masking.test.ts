import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import type {
  ReadableSpan,
  Span,
  SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import type { ExportResult } from '@opentelemetry/core';
import type { Attributes } from '@opentelemetry/api';
import type * as t from '@/types';
import { createLangfuseSpanProcessor } from '@/langfuseToolOutputTracing';
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
 * Drives the full export path: the redacting wrapper around the real
 * `LangfuseSpanProcessor` (no `@langfuse/otel` mocks) with the params
 * `getLangfuseSpanProcessorParams` builds under `metricsOnly`. The test
 * proves the registry's privacy wiring reaches the SDK's masking layer, that
 * run-correlation identity survives it, and that status and exception
 * content is redacted before export.
 */
describe('Langfuse privacy masking', () => {
  async function exportSpanWithPrivacy(
    privacy: t.LangfusePrivacyConfig,
    attributes: Attributes,
    {
      recordException,
      statusMessage,
    }: { recordException?: Error; statusMessage?: string } = {}
  ): Promise<ReadableSpan | undefined> {
    const exporter = new InMemoryExporter();
    const langfuse: t.LangfuseConfig = {
      publicKey: 'pk-privacy',
      secretKey: 'sk-privacy',
      baseUrl: 'https://langfuse.privacy',
      privacy,
    };
    const params = getLangfuseSpanProcessorParams(langfuse);
    if (params == null) {
      throw new Error(
        'expected span processor params for configured credentials'
      );
    }
    const processor = createLangfuseSpanProcessor(
      { ...params, exporter, shouldExportSpan: () => true },
      langfuse
    );
    const provider = new BasicTracerProvider({
      spanProcessors: [processor],
    });

    const span = provider
      .getTracer('privacy-masking-test')
      .startSpan('ChatGeneration') as Span;
    span.setAttributes(attributes);
    if (recordException != null) {
      span.recordException(recordException);
    }
    if (statusMessage != null) {
      span.setStatus({ code: 2, message: statusMessage });
    }
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

  it('preserves run-correlation identity from masked metadata', async () => {
    const span = await exportSpanWithPrivacy(
      { mode: 'metricsOnly' },
      {
        [LangfuseOtelSpanAttributes.TRACE_METADATA]: JSON.stringify({
          messageId: 'run-1',
          parentMessageId: 'run-0',
          agentId: 'research',
          agentName: 'Research Agent',
          tenantId: 'tenant-7',
          userNotes: 'internal note with content',
        }),
        [LangfuseOtelSpanAttributes.OBSERVATION_METADATA]: JSON.stringify({
          messageId: 'run-1',
          nested: { secret: 'value' },
        }),
      }
    );

    const expectedIdentity = JSON.stringify({
      messageId: 'run-1',
      parentMessageId: 'run-0',
      agentId: 'research',
      agentName: 'Research Agent',
    });
    expect(span?.attributes[LangfuseOtelSpanAttributes.TRACE_METADATA]).toBe(
      expectedIdentity
    );
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_METADATA]
    ).toBe(JSON.stringify({ messageId: 'run-1' }));
  });

  it('redacts status messages and exception content in metricsOnly', async () => {
    const span = await exportSpanWithPrivacy(
      { mode: 'metricsOnly' },
      {
        [LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]:
          'Invalid user value sk-secret',
      },
      {
        statusMessage: 'upstream rejected sk-secret',
        recordException: new Error('tool failed with sk-secret in args'),
      }
    );

    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
    ).toBe('[CONTENT REDACTED]');
    expect(span?.status.message).toBe('[CONTENT REDACTED]');
    expect(span?.status.code).toBe(2);
    const exceptionEvent = span?.events.find(
      (event) => event.name === 'exception'
    );
    expect(exceptionEvent?.attributes?.['exception.message']).toBe(
      '[CONTENT REDACTED]'
    );
    expect(exceptionEvent?.attributes?.['exception.stacktrace']).toBe(
      '[CONTENT REDACTED]'
    );
  });

  it('exports content unchanged in full mode', async () => {
    const span = await exportSpanWithPrivacy(
      { mode: 'full' },
      {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: 'plain prompt',
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'plain completion',
        [LangfuseOtelSpanAttributes.OBSERVATION_METADATA]: JSON.stringify({
          messageId: 'run-1',
        }),
      },
      {
        statusMessage: 'kept status message',
        recordException: new Error('kept exception'),
      }
    );

    expect(span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]).toBe(
      'plain prompt'
    );
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
    ).toBe('plain completion');
    expect(
      span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_METADATA]
    ).toBe(JSON.stringify({ messageId: 'run-1' }));
    expect(span?.status.message).toBe('kept status message');
    const exceptionEvent = span?.events.find(
      (event) => event.name === 'exception'
    );
    expect(exceptionEvent?.attributes?.['exception.message']).toBe(
      'kept exception'
    );
  });
});

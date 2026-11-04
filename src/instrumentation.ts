import { randomBytes } from 'node:crypto';
import { setLangfuseTracerProvider } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { context, ROOT_CONTEXT, createContextKey } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import type {
  IdGenerator,
  ReadableSpan,
  Span,
  SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  getLangfuseDestinationKey,
  getLangfuseSpanProcessorParams,
  registerLangfuseManagedSpan,
  registerLangfuseTraceAnchorSpan,
} from '@/langfuseSpanRegistry';
import {
  resolveLangfuseConfigForSpan,
  resolveLangfuseScopeAgentId,
  resolveLangfuseTraceAnchor,
  resolveTraceIdSeedForSpan,
} from '@/langfuseRuntimeScope';
import { createLangfuseSpanProcessor } from '@/langfuseToolOutputTracing';
import { createLibreChatTraceAttributes } from '@/langfuse';
import { traceIdFromSeed } from '@/langfuseRuntimeContext';
import { isPresent } from '@/utils/misc';

/**
 * Per-run seed for deterministic Langfuse trace ids. When a run opts in
 * (`LangfuseConfig.deterministicTraceId`), it executes its stream inside
 * `runWithTraceIdSeed(runId, ...)` from `./langfuseRuntimeContext`, and the
 * IdGenerator below derives the root trace id from that seed instead of a
 * random one. This lets external systems (e.g. a host app recording user
 * feedback after the fact) attach scores or observations to the trace by
 * regenerating the same id from the run/message id; no trace lookup required.
 * With no active seed it falls back to random ids, so default behavior is
 * unchanged.
 */
class SeededTraceIdGenerator implements IdGenerator {
  generateTraceId(): string {
    const seed = resolveTraceIdSeedForSpan(context.active());
    return isPresent(seed)
      ? traceIdFromSeed(seed)
      : randomBytes(16).toString('hex');
  }

  generateSpanId(): string {
    return randomBytes(8).toString('hex');
  }
}

let langfuseTracerProvider: BasicTracerProvider | undefined;
let langfuseRoutingSpanProcessor: RoutingLangfuseSpanProcessor | undefined;
const contextManagerProbeKey = createContextKey(
  'langfuse-context-manager-probe'
);

function hasActiveContextManager(): boolean {
  return context.with(
    ROOT_CONTEXT.setValue(contextManagerProbeKey, true),
    () => context.active().getValue(contextManagerProbeKey) === true
  );
}

export function ensureOpenTelemetryContextManager(): void {
  if (hasActiveContextManager()) {
    return;
  }

  const contextManager = new AsyncLocalStorageContextManager();
  contextManager.enable();
  if (!context.setGlobalContextManager(contextManager)) {
    contextManager.disable();
  }
}

/** Cache key for processor instances: the destination plus processor-level
 *  policy (`toolOutputTracing` and `privacy` are baked into each processor's
 *  redaction behavior), unlike the pure destination identity used for span
 *  parenting. */
function getLangfuseProcessorCacheKey(
  destinationKey: string,
  langfuse?: t.LangfuseConfig
): string {
  return JSON.stringify({
    destinationKey,
    mediaUploadEnabled: langfuse?.mediaUploadEnabled,
    toolOutputTracing: langfuse?.toolOutputTracing,
    privacy: langfuse?.privacy,
  });
}

class RoutingLangfuseSpanProcessor implements SpanProcessor {
  // Processors live for the process lifetime. LibreChat tenant Langfuse
  // destinations are expected to be a bounded admin-managed set, and shutdown
  // drains every cached processor when the provider is disposed.
  private readonly processors = new Map<string, SpanProcessor>();
  private readonly spanProcessors = new WeakMap<object, SpanProcessor>();

  ensureProcessor(langfuse?: t.LangfuseConfig): SpanProcessor | undefined {
    const params = getLangfuseSpanProcessorParams(langfuse);
    if (params == null) {
      return undefined;
    }
    return this.ensureProcessorForKey(
      getLangfuseProcessorCacheKey(getLangfuseDestinationKey(params), langfuse),
      params,
      langfuse
    );
  }

  private ensureProcessorForKey(
    processorCacheKey: string,
    params: LangfuseSpanProcessorParams,
    langfuse?: t.LangfuseConfig
  ): SpanProcessor {
    const existing = this.processors.get(processorCacheKey);
    if (existing != null) {
      return existing;
    }

    const processor = createLangfuseSpanProcessor(params, langfuse);
    this.processors.set(processorCacheKey, processor);
    return processor;
  }

  onStart(span: Span, parentContext: Context): void {
    const langfuse = resolveLangfuseConfigForSpan(parentContext);
    const params = getLangfuseSpanProcessorParams(langfuse);
    if (params == null) {
      return;
    }

    const destinationKey = getLangfuseDestinationKey(params);
    const processor = this.ensureProcessorForKey(
      getLangfuseProcessorCacheKey(destinationKey, langfuse),
      params,
      langfuse
    );
    registerLangfuseManagedSpan(span, destinationKey);
    const traceAnchor = resolveLangfuseTraceAnchor(parentContext);
    if (traceAnchor != null) {
      registerLangfuseTraceAnchorSpan(
        traceAnchor,
        span,
        destinationKey,
        resolveLangfuseScopeAgentId(parentContext)
      );
    }

    const librechatTraceAttributes = createLibreChatTraceAttributes(
      langfuse?.librechatTraceAttributes ?? {}
    );
    if (Object.keys(librechatTraceAttributes).length > 0) {
      span.setAttributes(librechatTraceAttributes);
    }

    this.spanProcessors.set(span, processor);
    processor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.spanProcessors.get(span)?.onEnd(span);
  }

  async forceFlush(): Promise<void> {
    await Promise.all(
      Array.from(this.processors.values(), (processor) =>
        processor.forceFlush()
      )
    );
  }

  async shutdown(): Promise<void> {
    await Promise.all(
      Array.from(this.processors.values(), (processor) => processor.shutdown())
    );
  }
}

export function initializeLangfuseTracing(
  langfuse?: t.LangfuseConfig
): BasicTracerProvider | undefined {
  const params = getLangfuseSpanProcessorParams(langfuse);
  if (params == null) {
    return undefined;
  }

  if (langfuseTracerProvider != null) {
    langfuseRoutingSpanProcessor?.ensureProcessor(langfuse);
    return langfuseTracerProvider;
  }

  ensureOpenTelemetryContextManager();
  langfuseRoutingSpanProcessor = new RoutingLangfuseSpanProcessor();
  langfuseRoutingSpanProcessor.ensureProcessor(langfuse);
  langfuseTracerProvider = new BasicTracerProvider({
    spanProcessors: [langfuseRoutingSpanProcessor],
    idGenerator: new SeededTraceIdGenerator(),
  });

  setLangfuseTracerProvider(langfuseTracerProvider);
  return langfuseTracerProvider;
}

export function initializeLangfuseTracingFromEnv():
  | BasicTracerProvider
  | undefined {
  return initializeLangfuseTracing();
}

initializeLangfuseTracingFromEnv();

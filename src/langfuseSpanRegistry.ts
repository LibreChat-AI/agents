import { createHash } from 'node:crypto';
import { context, trace } from '@opentelemetry/api';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Span, SpanContext } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
} from '@/langfuseConfig';
import { isPresent } from '@/utils/misc';

/**
 * Spans created through the Langfuse tracer provider, keyed by the export
 * destination they were routed to. Callback handlers use this to distinguish
 * safe ambient parents from spans a root observation must not inherit:
 *
 * - Foreign spans (a host's own OpenTelemetry instrumentation, e.g. HTTP
 *   server spans on the global provider) are never exported to Langfuse —
 *   inheriting one orphans the trace root (its root-observation input/output
 *   shaping is skipped), collapses concurrent runs inside one request
 *   context into a single merged trace, and bypasses the seeded
 *   deterministic trace id generator.
 * - Langfuse-managed spans bound to a *different* destination (another
 *   tenant's project) would leave the new trace dangling in its own
 *   destination while inheriting the other tenant's trace id.
 *
 * Only a managed span whose destination matches the starting run's resolved
 * destination is a safe parent — that is the sanctioned way for hosts to
 * group runs under their own Langfuse observations.
 */
const managedSpanDestinations = new WeakMap<Span, string>();

type AnchoredSpan = {
  destinationKey: string;
  spanContext: SpanContext;
};

type TraceAnchorState = {
  spans: Map<string, AnchoredSpan>;
  spanIds: Set<string>;
};

const ROOT_TRACE_ANCHOR = '';
const traceAnchorStates = new WeakMap<object, TraceAnchorState>();

export function registerLangfuseManagedSpan(
  span: Span,
  destinationKey: string
): void {
  managedSpanDestinations.set(span, destinationKey);
}

export function getLangfuseManagedSpanDestination(
  span: Span
): string | undefined {
  return managedSpanDestinations.get(span);
}

/** Captures the first exported observation for a run and for each agent lane. */
export function registerLangfuseTraceAnchorSpan(
  anchor: object,
  span: Span,
  destinationKey: string,
  agentId?: string
): void {
  let state = traceAnchorStates.get(anchor);
  if (state == null) {
    state = { spans: new Map<string, AnchoredSpan>(), spanIds: new Set() };
    traceAnchorStates.set(anchor, state);
  }
  const spanContext = span.spanContext();
  state.spanIds.add(spanContext.spanId);
  const key = agentId ?? ROOT_TRACE_ANCHOR;
  if (state.spans.has(key)) {
    return;
  }
  state.spans.set(key, { destinationKey, spanContext });
}

/**
 * Resolves the most specific safe parent available for auxiliary work. An
 * active observation wins so labels emitted inside a tool or reasoning step
 * stay beside their source; otherwise the run root anchors the label in the
 * same trace. An agent lane is the fallback when its overlay exports to a
 * different destination. Cross-destination parents are never returned.
 */
export function resolveLangfuseTraceAnchorParent(
  anchor: object | undefined,
  destinationKey: string | undefined,
  agentId?: string
): SpanContext | undefined {
  if (anchor == null || destinationKey == null) {
    return undefined;
  }
  const state = traceAnchorStates.get(anchor);
  if (state == null) {
    return undefined;
  }
  const agentSpan = agentId == null ? undefined : state.spans.get(agentId);
  const rootSpan = state.spans.get(ROOT_TRACE_ANCHOR);
  let anchoredSpan = rootSpan;
  if (anchoredSpan?.destinationKey !== destinationKey) {
    anchoredSpan = agentSpan;
  }
  if (anchoredSpan?.destinationKey !== destinationKey) {
    return undefined;
  }

  const activeSpan = trace.getSpan(context.active());
  if (
    activeSpan != null &&
    getLangfuseManagedSpanDestination(activeSpan) === destinationKey &&
    activeSpan.spanContext().traceId === anchoredSpan.spanContext.traceId &&
    state.spanIds.has(activeSpan.spanContext().spanId)
  ) {
    return activeSpan.spanContext();
  }
  return anchoredSpan.spanContext;
}

function resolveLangfuseEnvironment(
  langfuse?: t.LangfuseConfig
): string | undefined {
  const candidates = [
    langfuse?.environment,
    process.env.LANGFUSE_TRACING_ENVIRONMENT,
    process.env.NODE_ENV,
  ];
  for (const candidate of candidates) {
    if (candidate != null && candidate.trim() !== '') {
      return candidate.trim();
    }
  }
  return undefined;
}

function hasAdditionalHeaders(
  headers?: Record<string, string>
): headers is Record<string, string> {
  return headers != null && Object.keys(headers).length > 0;
}

export function getLangfuseSpanProcessorParams(
  langfuse?: t.LangfuseConfig
): LangfuseSpanProcessorParams | undefined {
  if (langfuse?.enabled === false) {
    return undefined;
  }
  const environment = resolveLangfuseEnvironment(langfuse);
  const additionalHeaders = hasAdditionalHeaders(langfuse?.additionalHeaders)
    ? { additionalHeaders: langfuse.additionalHeaders }
    : {};
  // metricsOnly suppresses media too: media payloads are content, so the
  // processor must not run its media create/upload/patch operations. Content
  // attributes themselves are redacted by the wrapping span processor, which
  // unlike the SDK `mask` callback knows which attribute it is handling.
  const metricsOnly = langfuse?.privacy?.mode === 'metricsOnly';
  const mediaUploadEnabled = metricsOnly ? false : langfuse?.mediaUploadEnabled;
  const contentPolicy = {
    ...(mediaUploadEnabled != null ? { mediaUploadEnabled } : {}),
  };
  if (hasLangfuseConfigCredentials(langfuse)) {
    return {
      publicKey: langfuse.publicKey,
      secretKey: langfuse.secretKey,
      ...(isPresent(langfuse.baseUrl) ? { baseUrl: langfuse.baseUrl } : {}),
      ...(isPresent(environment) ? { environment } : {}),
      ...contentPolicy,
      ...additionalHeaders,
    };
  }
  if (hasLangfuseEnvConfig()) {
    const baseUrl =
      langfuse?.baseUrl ??
      process.env.LANGFUSE_BASE_URL ??
      process.env.LANGFUSE_BASEURL;
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      ...(isPresent(baseUrl) ? { baseUrl } : {}),
      ...(isPresent(environment) ? { environment } : {}),
      ...contentPolicy,
      ...additionalHeaders,
    };
  }
  if (isPresent(langfuse?.baseUrl) && hasLangfuseEnvCredentials()) {
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      baseUrl: langfuse.baseUrl,
      ...(isPresent(environment) ? { environment } : {}),
      ...contentPolicy,
      ...additionalHeaders,
    };
  }
  return undefined;
}

function hashCacheKeyValue(value: string | undefined): string | undefined {
  return isPresent(value)
    ? createHash('sha256').update(value, 'utf8').digest('hex')
    : undefined;
}

/**
 * Order- and case-insensitive digest of the custom headers sent to a
 * destination, so header maps that differ only in key order or header-name
 * casing resolve to one destination instead of duplicating its exporter.
 * Hashed because these values are credentials (proxy tokens, gateway keys).
 * Absent and empty both yield `undefined`, keeping keys stable for the
 * overwhelmingly common no-headers case.
 */
function hashAdditionalHeaders(
  headers: Record<string, string> | undefined
): string | undefined {
  if (!hasAdditionalHeaders(headers)) {
    return undefined;
  }
  const normalized = Object.entries(headers)
    .map(([name, value]) => JSON.stringify([name.trim().toLowerCase(), value]))
    .sort();
  return hashCacheKeyValue(normalized.join('\n'));
}

/**
 * Identity of an export destination (project credentials + endpoint +
 * environment + custom headers) only. Processor-level policies like
 * `toolOutputTracing` are deliberately excluded: two spans exporting to the
 * same project under different redaction settings still share a destination
 * and may parent one another.
 *
 * Custom headers are included because a gateway may route on them, making two
 * otherwise-identical configs different projects. Treating them as part of the
 * destination keeps a run from inheriting a parent span bound elsewhere, and
 * keeps a rotated proxy credential from reusing the stale exporter.
 */
export function getLangfuseDestinationKey(
  params: LangfuseSpanProcessorParams
): string {
  return JSON.stringify({
    publicKey: params.publicKey,
    secretKeyHash: hashCacheKeyValue(params.secretKey),
    baseUrl: params.baseUrl,
    environment: params.environment,
    additionalHeadersHash: hashAdditionalHeaders(params.additionalHeaders),
  });
}

/** The export destination a run with this config resolves to, or `undefined`
 *  when no Langfuse destination is configured. */
export function resolveLangfuseDestinationKey(
  langfuse?: t.LangfuseConfig
): string | undefined {
  const params = getLangfuseSpanProcessorParams(langfuse);
  return params == null ? undefined : getLangfuseDestinationKey(params);
}

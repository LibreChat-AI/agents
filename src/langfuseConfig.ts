import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type * as t from '@/types';
import { parseBooleanEnv } from '@/utils/misc';

export const LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT = '[tool output redacted]';
export const LANGFUSE_CONTENT_REDACTION_TEXT = '[CONTENT REDACTED]';

function isPresent(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

export function normalizeToolName(name: string): string {
  return name.trim().toLowerCase();
}

export function hasLangfuseConfigCredentials(
  langfuse?: t.LangfuseConfig
): langfuse is t.LangfuseConfig & {
  publicKey: string;
  secretKey: string;
} {
  return (
    langfuse != null &&
    isPresent(langfuse.publicKey) &&
    isPresent(langfuse.secretKey)
  );
}

export function hasLangfuseEnvCredentials(): boolean {
  return (
    isPresent(process.env.LANGFUSE_SECRET_KEY) &&
    isPresent(process.env.LANGFUSE_PUBLIC_KEY)
  );
}

export function hasLangfuseEnvConfig(): boolean {
  return hasLangfuseEnvCredentials();
}

function normalizeToolNames(names: string[] | undefined): Set<string> {
  const normalized = new Set<string>();
  for (const name of names ?? []) {
    if (isPresent(name)) {
      normalized.add(normalizeToolName(name));
    }
  }
  return normalized;
}

function parseToolNames(value: string | undefined): string[] | undefined {
  if (!isPresent(value)) {
    return undefined;
  }

  return value
    .split(',')
    .map((name) => name.trim())
    .filter((name) => name !== '');
}

function getEnvToolOutputTracingEnabled(): boolean | undefined {
  const traceToolOutputs = parseBooleanEnv(
    process.env.LANGFUSE_TRACE_TOOL_OUTPUTS
  );
  if (traceToolOutputs != null) {
    return traceToolOutputs;
  }

  const redactToolOutputs = parseBooleanEnv(
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS
  );
  if (redactToolOutputs != null) {
    return !redactToolOutputs;
  }

  return parseBooleanEnv(process.env.LANGFUSE_TOOL_OUTPUT_TRACING_ENABLED);
}

function getEnvRedactedToolNames(): string[] | undefined {
  return (
    parseToolNames(process.env.LANGFUSE_REDACT_TOOL_OUTPUT_NAMES) ??
    parseToolNames(process.env.LANGFUSE_REDACT_TOOL_NAMES)
  );
}

function getEnvRedactionText(): string | undefined {
  return isPresent(process.env.LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT)
    ? process.env.LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    : undefined;
}

function getEnvToolNameMatchMode(): 'exact' | 'partial' | undefined {
  const mode = (
    process.env.LANGFUSE_REDACT_TOOL_OUTPUT_NAME_MATCH_MODE ??
    process.env.LANGFUSE_REDACT_TOOL_NAME_MATCH_MODE
  )
    ?.trim()
    .toLowerCase();
  if (mode === 'exact' || mode === 'partial') {
    return mode;
  }
  return undefined;
}

function hasEnvToolOutputTracingConfig(): boolean {
  return (
    getEnvToolOutputTracingEnabled() != null ||
    getEnvRedactedToolNames() != null ||
    getEnvRedactionText() != null ||
    getEnvToolNameMatchMode() != null
  );
}

function resolveToolOutputTracingEnabled(
  runConfig?: t.LangfuseToolOutputTracingConfig,
  agentConfig?: t.LangfuseToolOutputTracingConfig
): boolean {
  return (
    agentConfig?.enabled ??
    runConfig?.enabled ??
    getEnvToolOutputTracingEnabled() ??
    true
  );
}

function resolveRedactedToolNames(
  runConfig?: t.LangfuseToolOutputTracingConfig,
  agentConfig?: t.LangfuseToolOutputTracingConfig
): Set<string> {
  return normalizeToolNames([
    ...(getEnvRedactedToolNames() ?? []),
    ...(runConfig?.redactedToolNames ?? []),
    ...(agentConfig?.redactedToolNames ?? []),
  ]);
}

function resolveToolNameMatchMode(
  runConfig?: t.LangfuseToolOutputTracingConfig,
  agentConfig?: t.LangfuseToolOutputTracingConfig
): 'exact' | 'partial' {
  const modes = [
    getEnvToolNameMatchMode(),
    runConfig?.redactedToolNameMatchMode,
    agentConfig?.redactedToolNameMatchMode,
  ];
  return modes.includes('partial') ? 'partial' : 'exact';
}

export function hasToolOutputTracingConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): boolean {
  return (
    runLangfuse?.toolOutputTracing != null ||
    agentLangfuse?.toolOutputTracing != null ||
    hasEnvToolOutputTracingConfig()
  );
}

/**
 * Resolves the effective privacy policy. Privacy is a run-level deployment
 * policy, not a per-agent knob: shared spans (the root trace, conversation
 * payloads, activity-phase traces, parent generations embedding subagent
 * results) carry every agent's content, and a policy that only some of them
 * enforce would leak the others'. Agent overlays therefore cannot set it,
 * and one that asks for `metricsOnly` when the run has not fails closed
 * (export disabled) rather than export content the host asked to suppress.
 */
export function resolveLangfusePrivacyConfig(
  runLangfuse?: t.LangfuseConfig
): t.LangfusePrivacyConfig | undefined {
  return runLangfuse?.privacy;
}

/** Whether an agent overlay demands privacy the run does not provide. */
export function hasUnsupportedAgentPrivacy(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): boolean {
  return (
    agentLangfuse?.privacy?.mode === 'metricsOnly' &&
    runLangfuse?.privacy?.mode !== 'metricsOnly'
  );
}

export function resolveLangfuseContentRedactionText(
  privacy?: t.LangfusePrivacyConfig
): string {
  return isPresent(privacy?.redactionText)
    ? privacy.redactionText.trim()
    : LANGFUSE_CONTENT_REDACTION_TEXT;
}

export function resolveToolOutputTracingConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): ResolvedLangfuseToolOutputTracingConfig {
  const runConfig = runLangfuse?.toolOutputTracing;
  const agentConfig = agentLangfuse?.toolOutputTracing;

  return {
    enabled: resolveToolOutputTracingEnabled(runConfig, agentConfig),
    redactedToolNames: resolveRedactedToolNames(runConfig, agentConfig),
    redactedToolNameMatchMode: resolveToolNameMatchMode(runConfig, agentConfig),
    redactionText:
      agentConfig?.redactionText ??
      runConfig?.redactionText ??
      getEnvRedactionText() ??
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
  };
}

/**
 * Merges header maps case-insensitively, keeping the override's casing.
 *
 * A plain spread would keep both `X-Proxy-Token` and `x-proxy-token`, and
 * filling a fetch `Headers` from that record *appends* rather than replaces —
 * the exporter would send one comma-joined `run-token, agent-token` value, so
 * the agent override never cleanly wins and a gateway sees a malformed
 * credential. Matches the case-insensitive identity already used for the
 * destination key.
 */
function mergeAdditionalHeaders(
  base?: Record<string, string>,
  override?: Record<string, string>
): Record<string, string> | undefined {
  if (base == null && override == null) {
    return undefined;
  }

  const merged: Record<string, string> = { ...base };
  if (override == null) {
    return merged;
  }

  const baseKeyByLower = new Map<string, string>(
    Object.keys(merged).map((key) => [key.toLowerCase(), key])
  );
  for (const [key, value] of Object.entries(override)) {
    const lower = key.toLowerCase();
    const existingKey = baseKeyByLower.get(lower);
    if (existingKey != null && existingKey !== key) {
      delete merged[existingKey];
    }
    merged[key] = value;
    baseKeyByLower.set(lower, key);
  }
  return merged;
}

export function resolveLangfuseConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): t.LangfuseConfig | undefined {
  if (runLangfuse == null) {
    // No run config exists to carry a run-level policy, so the overlay's
    // privacy cannot be enforced either: strip it, failing closed when it
    // asked for metricsOnly.
    if (agentLangfuse?.privacy?.mode === 'metricsOnly') {
      return { ...agentLangfuse, privacy: undefined, enabled: false };
    }
    return agentLangfuse?.privacy == null
      ? agentLangfuse
      : { ...agentLangfuse, privacy: undefined };
  }
  if (agentLangfuse == null) {
    return runLangfuse;
  }

  const toolNodeTracing =
    runLangfuse.toolNodeTracing != null || agentLangfuse.toolNodeTracing != null
      ? {
        ...runLangfuse.toolNodeTracing,
        ...agentLangfuse.toolNodeTracing,
      }
      : undefined;
  const toolOutputTracing =
    runLangfuse.toolOutputTracing != null ||
    agentLangfuse.toolOutputTracing != null
      ? {
        ...runLangfuse.toolOutputTracing,
        ...agentLangfuse.toolOutputTracing,
      }
      : undefined;
  const metadata =
    runLangfuse.metadata != null || agentLangfuse.metadata != null
      ? {
        ...runLangfuse.metadata,
        ...agentLangfuse.metadata,
      }
      : undefined;
  const additionalHeaders = mergeAdditionalHeaders(
    runLangfuse.additionalHeaders,
    agentLangfuse.additionalHeaders
  );
  const librechatTraceAttributes =
    runLangfuse.librechatTraceAttributes != null ||
    agentLangfuse.librechatTraceAttributes != null
      ? {
        ...runLangfuse.librechatTraceAttributes,
        ...agentLangfuse.librechatTraceAttributes,
      }
      : undefined;
  const tags =
    runLangfuse.tags != null || agentLangfuse.tags != null
      ? [
        ...new Set([
          ...(runLangfuse.tags ?? []),
          ...(agentLangfuse.tags ?? []),
        ]),
      ]
      : undefined;
  return {
    ...runLangfuse,
    ...agentLangfuse,
    ...(metadata != null ? { metadata } : {}),
    ...(additionalHeaders != null ? { additionalHeaders } : {}),
    ...(librechatTraceAttributes != null ? { librechatTraceAttributes } : {}),
    ...(tags != null ? { tags } : {}),
    ...(toolNodeTracing != null ? { toolNodeTracing } : {}),
    ...(toolOutputTracing != null ? { toolOutputTracing } : {}),
    // Privacy is run-level: the overlay's copy above never takes effect.
    ...(runLangfuse.privacy != null
      ? { privacy: runLangfuse.privacy }
      : { privacy: undefined }),
    ...(hasUnsupportedAgentPrivacy(runLangfuse, agentLangfuse)
      ? { enabled: false }
      : {}),
  };
}

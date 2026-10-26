/** Host context preparation and trusted subagent execution identity are supported. */
export const SUBAGENT_CONTEXT_VERSION = 1;

/**
 * Anthropic direct API tool schema overhead multiplier.
 * Empirically calibrated against real MCP tool sets (29 tools).
 * Accounts for Anthropic's internal XML-like tool encoding plus
 * a ~300-token hidden tool-system preamble.
 */
export const ANTHROPIC_TOOL_TOKEN_MULTIPLIER = 2.6;

/**
 * Default tool schema overhead multiplier for all non-Anthropic providers.
 * Covers OpenAI function-calling format, Bedrock, and other providers.
 * Empirically calibrated at ~1.4× the raw JSON token count.
 */
export const DEFAULT_TOOL_TOKEN_MULTIPLIER = 1.4;

/**
 * Default ceiling on cooperative stream seals per run. Each seal costs one
 * extra superstep, so this also sizes the recursion-limit headroom a
 * preemption-enabled run reserves.
 */
export const DEFAULT_MAX_SEALS = 8;
/**
 * How long a preempt request waits for the turn to become sealable before it
 * is allowed to discard the turn instead.
 *
 * Exists to keep a restart from stealing a seal that was about to happen: a
 * model streaming reasoning is usually moments from its first text, and
 * keeping that answer beats throwing the reasoning away. Only a request that
 * outlives the window — a genuinely long thinking stretch — converts.
 *
 * Applies to a silent provider too. The accumulation a restart is judged
 * against is what the consumer has seen, and the stream buffers a chunk ahead
 * of it — so "nothing accumulated" can also mean "the first chunk is in
 * flight", and converting on it would discard that chunk unseen.
 */
export const DEFAULT_PREEMPT_RESTART_GRACE_MS = 2_000;

/**
 * Default ceiling on Stop-hook continuations within one Run. A blocking Stop
 * hook can keep a naturally terminal run warm by injecting another user turn;
 * the ceiling prevents a faulty hook or continuously arriving input from
 * keeping one processStream call alive forever.
 */
export const DEFAULT_MAX_STOP_CONTINUATIONS = 8;

/**
 * Per-hook timeout for `PreemptBoundary`, deliberately far above
 * `DEFAULT_HOOK_TIMEOUT_MS`. A host drain has already popped its queue and
 * persisted content by the time this fires, so a timeout does not cancel the
 * work — it only loses the messages the run was about to inject.
 */
export const PREEMPT_BOUNDARY_HOOK_TIMEOUT_MS = 120_000;

/**
 * Default LangGraph recursion limit for a run. Callers may override it via
 * the stream config; a preemption-enabled run adds its seal budget on top.
 */
export const DEFAULT_RECURSION_LIMIT = 50;

/** Stable runtime names used for agent workflow observations. */
export const STANDARD_GRAPH_RUN_NAME = 'AgentGraph';
export const MULTI_AGENT_GRAPH_RUN_NAME = 'MultiAgentGraph';
export const AGENT_MODEL_CALL_RUN_NAME = 'AgentModelCall';
export const ACTIVITY_LABEL_RUN_NAME = 'StepLabel';
export const REASONING_LABEL_RUN_NAME = 'ReasoningLabel';
export const ACTIVITY_PHASE_RUN_NAME = 'MultiStepLabel';
export const ACTIVITY_PHASE_LABEL_RUN_NAME = 'MultiStepLabelGeneration';

/** Shared admission and rendering bounds for compaction navigation hints. */
export const COMPACTION_SEMANTIC_INDEX_LIMITS = Object.freeze({
  maxInputEntries: 256,
  maxEntries: 64,
  maxEntryChars: 512,
  maxTotalChars: 4_096,
  maxInputTextChars: 4_096,
  maxIdentityChars: 512,
  maxSourceContentIndex: 4_095,
} as const);

/**
 * Whether this build enforces `LangfuseConfig.privacy` (content redaction
 * and media suppression on exported traces). Hosts gate on it to fail
 * closed: with an older runtime that ignores the field, a privacy mode must
 * disable export rather than silently send unmasked content.
 */
export const LANGFUSE_PRIVACY_ENFORCEMENT_SUPPORTED = true;

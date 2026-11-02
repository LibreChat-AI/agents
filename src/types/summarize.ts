import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import type { ProviderName, ContextBudgetMode } from '@/types/llm';
import type { SummaryContentBlock } from '@/types/stream';

export type SummarizationTrigger = {
  type:
    | 'token_ratio'
    | 'remaining_tokens'
    | 'messages_to_refine'
    | (string & {});
  value: number;
};

/**
 * Controls how much recent context is preserved verbatim during compaction.
 * User-turn boundaries are preferred. Under context pressure, older closed
 * tool units inside an otherwise indivisible turn may be summarized while a
 * token-priced recent tail is retained. A lone user payload stays intact.
 */
export type RetainRecentConfig = {
  /**
   * Maximum number of recent user-led turns to keep in the tail. A turn begins
   * at a user-authored HumanMessage and includes every following AIMessage and
   * tool result up to the next user-authored HumanMessage. Provider-native
   * HumanMessages containing only tool results remain in the current turn.
   * Set to `0` to disable the recency window (legacy behavior: summarize
   * everything). Defaults to `2`.
   */
  turns?: number;
  /**
   * Optional retained-recent token budget. Older turns are added whole only
   * while cumulative tokens stay below the cap. If a tool-heavy history has
   * no compactable turn-level head, this is also the minimum recent tail kept
   * behind a pairing-balanced intra-turn cut. When omitted, that fallback
   * retains 16% of the configured context window.
   */
  tokens?: number;
};

export type CompactionSemanticIndexStatus = 'committed' | 'pending';

type CompactionSemanticIndexEntryBase = {
  /** Persisted message that owns the indexed content. */
  sourceMessageId: string;
  /** Zero-based content-part index within the persisted source message. */
  sourceContentIndex: number;
  /** Monotonic host revision for this logical entry. */
  revision: number;
  /** Only committed entries may guide compaction. */
  status: CompactionSemanticIndexStatus;
  /** User-visible semantic guidance. Hidden reasoning must never be supplied. */
  text: string;
  /** Omits the entry entirely when host policy redacts its source. */
  redacted?: boolean;
};

export type CompactionToolSemanticIndexEntry =
  CompactionSemanticIndexEntryBase & {
    type: 'tool_intent' | 'tool_outcome';
    toolCallId: string;
  };

export type CompactionActivitySemanticIndexEntry =
  CompactionSemanticIndexEntryBase & {
    type: 'activity_phase';
  };

export type CompactionReasoningSemanticIndexEntry =
  CompactionSemanticIndexEntryBase & {
    type: 'reasoning_label';
    /** Stable identity shared by every user-visible label revision. */
    reasoningStepId: string;
  };

/**
 * Source-addressed navigation hints for the compaction model. Entries remain
 * advisory: raw messages are always sent and remain authoritative.
 */
export type CompactionSemanticIndexEntry =
  | CompactionToolSemanticIndexEntry
  | CompactionActivitySemanticIndexEntry
  | CompactionReasoningSemanticIndexEntry;

export type CompactionSemanticIndex = readonly CompactionSemanticIndexEntry[];

/** Serializable continuation state for a bounded compaction semantic index. */
export type CompactionSemanticIndexSnapshot = {
  entries: CompactionSemanticIndex;
  /** Cumulative entries supplied before validation and bounded retention. */
  providedEntryCount: number;
};

export type SummarizationConfig = {
  provider?: ProviderName;
  model?: string;
  parameters?: Record<string, unknown>;
  prompt?: string;
  updatePrompt?: string;
  trigger?: SummarizationTrigger;
  maxSummaryTokens?: number;
  /** maxContextTokens is an input ceiling by default; use window for a full model window. */
  contextBudgetMode?: ContextBudgetMode;
  /** Fraction of the token budget reserved as headroom (0–1). Defaults to 0.05. */
  reserveRatio?: number;
  /**
   * Recent-message preservation policy.  When unset, defaults to
   * `{ turns: 2 }` so the last two user-led turns are kept verbatim
   * while older content is summarized.  Setting `{ turns: 0 }` reverts
   * to the legacy behavior of summarizing every message.
   */
  retainRecent?: RetainRecentConfig;
};

export interface SummarizeResult {
  text: string;
  tokenCount: number;
  model?: string;
  provider?: string;
}

export interface SummarizationNodeInput {
  remainingContextTokens: number;
  agentId: string;
  /** Pruned, masked provider history from the turn requesting compaction. */
  windowMessages?: BaseMessage[];
  /**
   * Why the detour was requested.
   *
   * - `trigger` (default): the configured summarization trigger fired during
   *   the pre-call budget check.
   * - `overflow`: the provider rejected the prompt as too large and the run
   *   is compacting to recover. When summarization is not enabled, this
   *   variant performs no model call — the corrected budget alone is what the
   *   retry needs.
   * - `manual`: the host asked for a summary outright (a summarize-only run).
   *   Uses the same bounded recent-tail policy as automatic compaction.
   */
  reason?: 'trigger' | 'overflow' | 'manual';
  /**
   * Whether an overflow recovery may spend a summarization model call.
   *
   * The first recovery deliberately does not: re-pruning against the
   * corrected budget raises context pressure, which drives the pruner's
   * existing tool-output compression and masking. That is cheaper, needs no
   * model call, and cannot lose message content the way a summary can. Only
   * when deterministic compression proves insufficient does the next attempt
   * allow the summarizer to run.
   */
  allowSummarization?: boolean;
}

export interface SummarizeStartEvent {
  agentId: string;
  provider: string;
  model?: string;
  messagesToRefineCount: number;
  /** Which summarization cycle this is (1-based, increments each time summarization fires) */
  summaryVersion: number;
  /** Committed, source-valid semantic hints included in the request. */
  semanticIndexEntryCount?: number;
  /** Serialized semantic-index characters included in the request. */
  semanticIndexCharCount?: number;
}

export interface SummarizeDeltaEvent {
  id: string;
  delta: {
    summary: SummaryContentBlock;
  };
}

export type SummarizationErrorCode =
  | 'provider_overloaded'
  | 'rate_limited'
  | 'context_exceeded'
  | 'timeout'
  | 'unknown';

export interface SummarizationAttemptUsage {
  provider: ProviderName;
  model?: string;
  usage: Partial<UsageMetadata>;
}

export interface SummarizeCompleteEvent {
  id: string;
  agentId: string;
  summary?: SummaryContentBlock;
  error?: string;
  errorCode?: SummarizationErrorCode;
  attemptUsage?: readonly SummarizationAttemptUsage[];
}

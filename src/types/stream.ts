// src/types/stream.ts
import type {
  MessageContentImageUrl,
  MessageContentText,
  ToolMessage,
  BaseMessage,
} from '@langchain/core/messages';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { LLMResult, Generation } from '@langchain/core/outputs';
import type { Command } from '@langchain/langgraph';
import type Anthropic from '@anthropic-ai/sdk';
import type { AnthropicContentBlock } from '@/llm/anthropic/types';
import type { AssistantTextPhase } from '@/types/assistantPhase';
import type { SummarizeCompleteEvent } from '@/types/summarize';
import type { NativeMediaReference } from '@/llm/google/native';
import type { ToolEndEvent } from '@/types/tools';
import { StepTypes, ContentTypes, GraphEvents } from '@/common/enum';

export type HandleLLMEnd = (
  output: LLMResult,
  runId: string,
  parentRunId?: string,
  tags?: string[]
) => void;

export type MetadataAggregatorResult = {
  handleLLMEnd: HandleLLMEnd;
  collected: Record<string, unknown>[];
};

export type StreamGeneration = Generation & {
  text?: string;
  message?: BaseMessage;
};

/** Event names are of the format: on_[runnable_type]_(start|stream|end).

Runnable types are one of:

llm - used by non chat models
chat_model - used by chat models
prompt -- e.g., ChatPromptTemplate
tool -- LangChain tools
chain - most Runnables are of this type
Further, the events are categorized as one of:

start - when the runnable starts
stream - when the runnable is streaming
end - when the runnable ends
start, stream and end are associated with slightly different data payload.

Please see the documentation for EventData for more details. */
export type EventName = string;

export type RunStepStatus =
  | 'in_progress'
  | 'completed'
  | 'cancelled'
  | 'failed';

export type RunStep = {
  type: StepTypes;
  /** Epoch ms when the step was dispatched. */
  created_at?: number;
  /**
   * Lifecycle status; terminal values are stamped when the step closes.
   * Invariant (enforced by `closeRunStep`, not the type, to stay wire-compatible
   * with the OpenAI Assistants shape): a terminal status sets exactly one
   * matching `*_at` field; first close wins and `cancelled`/`failed` are
   * immutable once stamped.
   */
  status?: RunStepStatus;
  /** Epoch ms when the step closed with status `completed`. */
  completed_at?: number;
  /** Epoch ms when the step closed with status `cancelled` (abort/halt). */
  cancelled_at?: number;
  /** Epoch ms when the step closed with status `failed`. */
  failed_at?: number;
  id: string; // #new
  runId?: string; // #new
  agentId?: string; // #new - tracks which agent this step belongs to
  /**
   * Opaque positive safe-integer identifier for parallel execution.
   * Agents with the same groupId should be rendered together.
   * Consumers must use content indexes, not groupId ordering, for execution order.
   * undefined means the agent runs sequentially (not part of a parallel group).
   *
   * Example for: researcher -> [analyst1, analyst2, analyst3] -> summarizer
   * - researcher: undefined (sequential)
   * - analyst1, analyst2, analyst3: the same groupId (parallel group)
   * - summarizer: undefined (sequential)
   */
  groupId?: number; // #new
  index: number; // #new
  stepIndex?: number; // #new
  stepDetails: StepDetails;
  summary?: SummaryContentBlock;
  usage?: null | object;
  // {
  // Define usage structure if it's ever non-null
  // prompt_tokens: number; // #new
  // completion_tokens: number; // #new
  // total_tokens: number; // #new
  // };
};

/** Minimal durable lifecycle state needed to continue open run steps. */
export interface RunStepResumeEntry {
  step: RunStep;
  pendingToolCallIds: string[];
  latestCompletionAt?: number;
  openMessageStep: boolean;
}

/** SDK-private state persisted in LangGraph checkpoints for process-safe resume. */
export interface RunStepResumeState {
  version: 1;
  revision: number;
  nextIndex: number;
  /** Warm terminal continuations already admitted for this execution. */
  stopContinuationCount?: number;
  /** Identifies the fresh execution that owns this continuation lifecycle. */
  stopContinuationExecutionId?: string;
  /** Distinguishes LangGraph event keys when a warm continuation restarts steps. */
  streamSegment?: number;
  toolCallSteps: Array<{ toolCallId: string; stepId: string }>;
  steps: RunStepResumeEntry[];
}

/**
 * Represents a run step delta i.e. any changed fields on a run step during
 * streaming.
 */
export interface RunStepDeltaEvent {
  /**
   * The identifier of the run step, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * The delta containing the fields that have changed on the run step.
   */
  delta: ToolCallDelta;
}

/**
 * Terminal signal for a run step, emitted exactly once per step when it
 * finishes (`completed`), is aborted/halted (`cancelled`), or the run errors
 * (`failed`). The `id` is top-level so callback echoes dedupe like other
 * step-scoped events.
 */
export interface RunStepClosedEvent {
  id: string;
  index: number;
  type: StepTypes;
  status: Exclude<RunStepStatus, 'in_progress'>;
  /** Epoch ms when the step was dispatched, when known. */
  created_at?: number;
  /** Epoch ms when the step reached its terminal status. */
  closed_at: number;
  runId?: string;
  agentId?: string;
  groupId?: number;
  stepIndex?: number;
}

export type RecordStepCompletionOptions = {
  /** The completing tool call, when the step tracks pending completions. */
  toolCallId?: string;
  metadata?: Record<string, unknown>;
  /**
   * Producer-stamped completion time (epoch ms). Carried through so a slow
   * host completion handler cannot inflate the recorded step duration.
   */
  at?: number;
};

export type RunStepCloseOptions = {
  /** Epoch ms for the terminal stamp; defaults to `Date.now()` at close time. */
  at?: number;
  metadata?: Record<string, unknown>;
};

export type StepDetails = MessageCreationDetails | ToolCallsDetails;

export type SummaryCompleted = {
  type: 'summary';
  summary: SummaryContentBlock;
};

export type StepCompleted = ToolCallCompleted | SummaryCompleted;

export type MessageCreationDetails = {
  type: StepTypes.MESSAGE_CREATION;
  message_creation: {
    message_id: string;
    /** Content lane announced before its first delta. */
    content_type?: ContentTypes.TEXT | ContentTypes.THINK;
    /** Provider-authored assistant text phase, when available. */
    phase?: AssistantTextPhase;
  };
};

export type ToolEndData = {
  input: string | Record<string, unknown>;
  output?: ToolMessage | Command;
};
export type ToolErrorData = {
  id: string;
  name: string;
  error?: Error;
} & Pick<ToolEndData, 'input'>;
export type ToolEndCallback = (
  data: ToolEndData,
  metadata?: Record<string, unknown>
) => Promise<void>;

export type ProcessedToolCall = {
  name: string;
  args: string | Record<string, unknown>;
  id: string;
  output: string;
  progress: number;
  /**
   * Settled label for the call, resolved from the tool-supplied
   * `outcome`/`outcome_patch` result fields against the model-authored
   * `intent` arg. Present ONLY when the tool authored one.
   *
   * When absent, display the `intent` arg unchanged — do NOT rewrite its
   * tense. A gerund→past-tense rewrite can only be a closed list of English
   * verbs, so it never fires for the non-English labels this feature expects
   * and fires for some sibling calls but not others within one group.
   * Completion belongs to UI state (the shimmer stopping, the icon settling),
   * which is language-neutral and always consistent.
   */
  outcome?: string;
};

export type ProcessedContent = {
  type: ContentType;
  text?: string;
  tool_call?: ProcessedToolCall;
};

export type ToolCallCompleted = {
  type: 'tool_call';
  tool_call: ProcessedToolCall;
};

export type ToolCompleteEvent = ToolCallCompleted & {
  /** The Step Id of the Tool Call */
  id: string;
  /** The content index of the tool call */
  index: number;
  type: 'tool_call';
  /** Epoch ms when this tool call's completion was dispatched. */
  completed_at?: number;
};

export type ToolCallsDetails = {
  type: StepTypes.TOOL_CALLS;
  tool_calls?: AgentToolCall[]; // #new
};

export type ToolCallDelta = {
  type: StepTypes;
  tool_calls?: ToolCallChunk[]; // #new
  summary?: SummaryContentBlock;
  auth?: string;
  expires_at?: number;
};

export type AgentToolCall =
  | {
      id: string; // #new
      type: 'function'; // #new
      function: {
        name: string; // #new
        arguments: string | object; // JSON string // #new
      };
    }
  | ToolCall;

export interface ExtendedMessageContent {
  type?: string;
  text?: string;
  input?: string;
  /** Tool-call arguments on a v1 standard-content `tool_call` block. */
  args?: ToolCallPart['args'];
  index?: string | number;
  id?: string;
  name?: string;
}

export type AgentUpdate = {
  type: ContentTypes.AGENT_UPDATE;
  agent_update: {
    index: number;
    runId: string;
    agentId: string;
  };
};

/**
 * Represents a message delta i.e. any changed fields on a message during
 * streaming.
 */
export interface MessageDeltaEvent {
  /**
   * The identifier of the message, which can be referenced in API endpoints.
   */
  id: string;

  /**
   * The delta containing the fields that have changed on the Message.
   */
  delta: MessageDelta;
}

/**
 * The delta containing the fields that have changed on the Message.
 */
export interface MessageDelta {
  /**
   * The content of the message in array of text and/or images.
   */
  content?: MessageContentComplex[];
  /**
   * The tool call ids associated with the message.
   */
  tool_call_ids?: string[];
}

/**
 * Represents a reasoning delta i.e. any changed fields on a message during
 * streaming.
 */
export interface ReasoningDeltaEvent {
  /**
   * The identifier of the message, which can be referenced in API endpoints.
   */
  id: string;

  /**
   * The delta containing the fields that have changed.
   */
  delta: ReasoningDelta;
}

/**
 * The reasoning delta containing the fields that have changed on the Message.
 */
export interface ReasoningDelta {
  /**
   * The content of the message in array of text and/or images.
   */
  content?: MessageContentComplex[];
}

export type MessageDeltaUpdate = {
  native_media?: NativeMediaReference;
  type: ContentTypes.TEXT;
  text: string;
  tool_call_ids?: string[];
  /** Provider-supplied source citations, accumulated across deltas. */
  citations?: Anthropic.TextCitation[];
};
export type ReasoningDeltaUpdate = { type: ContentTypes.THINK; think: string };

export type ContentType =
  | 'text'
  | 'image_url'
  | 'tool_call'
  | 'think'
  | 'summary'
  | string;

export type ReasoningContentText = {
  type: ContentTypes.THINK;
  think: string;
};

export type SummaryBoundary = {
  messageId: string;
  contentIndex: number;
};

/**
 * Semantic extent of a summary: the first source message compaction retained
 * verbatim, meaning everything before it is covered. Distinct from `boundary`,
 * which records where the block was emitted — a retained recency tail sits
 * *before* the block's own position, so position alone cannot say what the
 * summary replaced.
 *
 * Anchored to the retained side rather than the covered side so that a source
 * message expanding into several messages (a steer splits an assistant entry
 * into pre-steer, steer, and post-steer entries sharing one ID) stays whole:
 * such a message is the retained anchor and survives intact.
 */
export type SummaryCoverage = {
  retainedFromMessageId: string;
};

export type SummaryContentBlock = {
  type: ContentTypes.SUMMARY;
  content?: MessageContentComplex[];
  /** Injection budget: provider output-token space when usage was reported, plus
   *  the wrapper added at injection time. Not comparable with per-message counts
   *  such as `indexTokenCountMap`, which are in the consumer's own tokenizer. */
  tokenCount?: number;
  coverage?: SummaryCoverage;
  boundary?: SummaryBoundary;
  summaryVersion?: number;
  model?: string;
  provider?: string;
  createdAt?: string;
};

/** Vertex AI / Google Common - Reasoning Content Block Format */
export type GoogleReasoningContentText = {
  type: ContentTypes.REASONING;
  reasoning: string;
};

/** Anthropic's Reasoning Content Block Format */
export type ThinkingContentText = {
  type: ContentTypes.THINKING;
  index?: number;
  signature?: string;
  thinking?: string;
};

/** Bedrock's Reasoning Content Block Format */
export type BedrockReasoningContentText = {
  type: ContentTypes.REASONING_CONTENT;
  index?: number;
  reasoningText: { text?: string; signature?: string };
};

/**
 * A call to a tool.
 */
export type ToolCallPart = {
  /** Type ("tool_call") according to Assistants Tool Call Structure */
  type: ContentTypes.TOOL_CALL;
  /** The name of the tool to be called */
  name?: string;
  /** The arguments to the tool call */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args?: string | Record<string, any>;

  /** If provided, an identifier associated with the tool call */
  id?: string;
  /** If provided, the output of the tool call */
  output?: ToolResultContent['content'];
  /**
   * Tool-authored settled label for the call (see `ProcessedToolCall.outcome`),
   * preserved through aggregation so it survives persistence/reload.
   */
  outcome?: string;
  /** Auth URL */
  auth?: string;
  /** Expiration time */
  expires_at?: number;
};

export type ToolCallContent = {
  type: ContentTypes.TOOL_CALL;
  tool_call?: ToolCallPart;
};

export type ToolResultContent = {
  content:
    | string
    | Record<string, unknown>
    | Array<string | Record<string, unknown>>
    | AnthropicContentBlock[];
  type: 'tool_result' | 'web_search_result' | 'web_search_tool_result';
  tool_use_id?: string;
  input?: string | Record<string, unknown>;
  index?: number;
};

export type MessageContentComplex = (
  | ToolResultContent
  | ThinkingContentText
  | SummaryContentBlock
  | AgentUpdate
  | ToolCallContent
  | ReasoningContentText
  | MessageContentText
  | MessageContentImageUrl
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | (Record<string, any> & {
      type?: 'text' | 'image_url' | 'think' | 'thinking' | string;
    })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | (Record<string, any> & {
      type?: never;
    })
) & {
  native_media?: NativeMediaReference;
  /** Open Responses-compatible semantic phase for assistant text. */
  phase?: AssistantTextPhase;
  /** LangChain standard-content form of provider-specific block fields. */
  extras?: { phase?: AssistantTextPhase } & Record<string, unknown>;
  tool_call_ids?: string[];
  // Optional agentId for parallel execution attribution
  agentId?: string;
  // Optional groupId for parallel group attribution
  groupId?: number;
};

export interface TMessage {
  role?: string;
  content?: MessageContentComplex[] | string;
  [key: string]: unknown;
}

export type TPayload = Array<Partial<TMessage>>;

export type SummarizeDeltaData = {
  id: string;
  delta: {
    summary: SummaryContentBlock;
  };
};

export type ContentAggregator = ({
  event,
  data,
}: {
  event: GraphEvents;
  data:
    | RunStep
    | AgentUpdate
    | MessageDeltaEvent
    | ReasoningDeltaEvent
    | RunStepDeltaEvent
    | SummarizeDeltaData
    | SummarizeCompleteEvent
    | { result: ToolEndEvent };
}) => void;
export type ContentAggregatorResult = {
  stepMap: Map<string, RunStep | undefined>;
  contentParts: Array<MessageContentComplex | undefined>;
  aggregateContent: ContentAggregator;
};

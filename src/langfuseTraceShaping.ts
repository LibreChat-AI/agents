import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import {
  Constants,
  STANDARD_GRAPH_RUN_NAME,
  MULTI_AGENT_GRAPH_RUN_NAME,
  AGENT_MODEL_CALL_RUN_NAME,
  ACTIVITY_LABEL_RUN_NAME,
  REASONING_LABEL_RUN_NAME,
  ACTIVITY_PHASE_RUN_NAME,
  ACTIVITY_PHASE_LABEL_RUN_NAME,
} from '@/common';
import { LANGFUSE_OPERATION_METADATA_KEY } from '@/langfuseOperation';

const LANGGRAPH_START_NODE = '__start__';
const LANGGRAPH_RUN_NAME = 'LangGraph';
const ANONYMOUS_LAMBDA_NAME = 'RunnableLambda';
const RUNNABLE_SEQUENCE_NAME = 'RunnableSequence';
const LANGGRAPH_AGENT_NODE_PREFIX = 'agent=';
const LANGGRAPH_TOOL_NODE_PREFIX = 'tools=';
const AGENT_NODE_SPAN_NAME = 'agent';
const TOOL_DISPATCH_SPAN_NAME = 'tool-dispatch';
const GENERATION_SPAN_NAME = 'llm';
const ROOT_OBSERVATION_TYPE = 'agent';
const CHAIN_OBSERVATION_TYPE = 'chain';
const TOOL_OBSERVATION_TYPE = 'tool';
const AGENT_TRACE_TAG = 'agent';
const TITLE_TRACE_TAG = 'title';
const ACTIVITY_LABEL_TRACE_TAG = 'activity-label';
const REASONING_LABEL_TRACE_TAG = 'reasoning-label';
const ACTIVITY_PHASE_TRACE_TAG = 'activity-phase';
const EPHEMERAL_AGENT_SENDER_SEPARATOR = '___';
const EPHEMERAL_AGENT_INDEX_SEPARATOR = '____';
const DEPRECATED_TRACE_INPUT_ATTRIBUTE = 'langfuse.trace.input';
const DEPRECATED_TRACE_OUTPUT_ATTRIBUTE = 'langfuse.trace.output';
const OBSERVATION_METADATA_LANGGRAPH_NODE = 'langgraph_node';
const OBSERVATION_METADATA_OPERATION = LANGFUSE_OPERATION_METADATA_KEY;
const OBSERVATION_METADATA_COMPACTION_SEMANTIC_INDEX_ENTRIES = `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.compaction_semantic_index_entries`;
const COMPACTION_SEMANTIC_INDEX_OPEN = '<compaction-semantic-index>';
const COMPACTION_SEMANTIC_INDEX_CLOSE = '</compaction-semantic-index>';
const REDACTED_COMPACTION_SEMANTIC_INDEX =
  '<compaction-semantic-index redacted="true" />\n\n';

/**
 * Returns an observation metadata value. `@langfuse/tracing` stores each
 * metadata value on the span as JSON, strings included.
 */
export function getObservationMetadataAttribute(
  attributes: Record<string, unknown>,
  key: string
): unknown {
  const value =
    attributes[`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.${key}`];
  if (typeof value !== 'string') {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

type MutableSpan = ReadableSpan & {
  name: string;
  attributes: Record<string, unknown>;
};

type SerializedToolCall = {
  name: string;
  args: unknown;
  id?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function parseAttributeValue(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return value;
  }
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

type CompactionSemanticIndexRedaction = {
  value: unknown;
  redacted: boolean;
};

function redactCompactionSemanticIndexText(
  value: string
): CompactionSemanticIndexRedaction {
  if (!value.startsWith(COMPACTION_SEMANTIC_INDEX_OPEN)) {
    return { value, redacted: false };
  }
  const start = 0;
  const close = value.indexOf(COMPACTION_SEMANTIC_INDEX_CLOSE, start);
  if (close < 0) {
    return { value, redacted: false };
  }
  let end = close + COMPACTION_SEMANTIC_INDEX_CLOSE.length;
  while (end < value.length && /\s/.test(value[end])) {
    end++;
  }
  return {
    value:
      value.slice(0, start) +
      REDACTED_COMPACTION_SEMANTIC_INDEX +
      value.slice(end),
    redacted: true,
  };
}

function redactCompactionSemanticIndexValue(
  value: unknown
): CompactionSemanticIndexRedaction {
  if (typeof value === 'string') {
    return redactCompactionSemanticIndexText(value);
  }
  if (Array.isArray(value)) {
    for (let index = value.length - 1; index >= 0; index--) {
      const nested = redactCompactionSemanticIndexValue(value[index]);
      if (nested.redacted) {
        const result = [...value];
        result[index] = nested.value;
        return { value: result, redacted: true };
      }
    }
    return { value, redacted: false };
  }
  if (!isRecord(value)) {
    return { value, redacted: false };
  }
  const entries = Object.entries(value);
  for (let index = entries.length - 1; index >= 0; index--) {
    const [key, nestedValue] = entries[index];
    const nested = redactCompactionSemanticIndexValue(nestedValue);
    if (nested.redacted) {
      return {
        value: { ...value, [key]: nested.value },
        redacted: true,
      };
    }
  }
  return { value, redacted: false };
}

function redactCompactionSemanticIndexInput(span: MutableSpan): void {
  const entryCount =
    span.attributes[
      OBSERVATION_METADATA_COMPACTION_SEMANTIC_INDEX_ENTRIES
    ];
  const numericEntryCount = Number(entryCount);
  if (!Number.isFinite(numericEntryCount) || numericEntryCount <= 0) {
    return;
  }
  const inputKey = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
  const input = span.attributes[inputKey];
  if (input == null) {
    return;
  }
  const parsed = parseAttributeValue(input);
  const redaction = redactCompactionSemanticIndexValue(parsed);
  if (!redaction.redacted) {
    return;
  }
  span.attributes[inputKey] =
    typeof redaction.value === 'string' && parsed === input
      ? redaction.value
      : JSON.stringify(redaction.value);
}

function getMessageArray(
  value: unknown
): Record<string, unknown>[] | undefined {
  if (Array.isArray(value)) {
    const records = value.filter(isRecord);
    return records.length > 0 ? records : undefined;
  }
  if (!isRecord(value)) {
    return undefined;
  }
  return (
    getMessageArray(value.messages) ??
    getMessageArray(value.input) ??
    getMessageArray(value.output)
  );
}

function getMessageRole(message: Record<string, unknown>): string | undefined {
  const id = message.id;
  if (Array.isArray(id)) {
    const className = id[id.length - 1];
    if (typeof className === 'string') {
      if (className.includes('Human')) {
        return 'user';
      }
      if (className.includes('AI')) {
        return 'assistant';
      }
      if (className.includes('System')) {
        return 'system';
      }
      if (className.includes('Tool')) {
        return 'tool';
      }
    }
  }
  const rawRole =
    message.type ?? message._type ?? message.role ?? message.sender;
  if (typeof rawRole !== 'string') {
    const kwargs = message.kwargs;
    return isRecord(kwargs) ? getMessageRole(kwargs) : undefined;
  }
  const normalized = rawRole.toLowerCase();
  if (normalized === 'human') {
    return 'user';
  }
  if (normalized === 'ai') {
    return 'assistant';
  }
  return normalized;
}

function getMessageText(message: Record<string, unknown>): string | undefined {
  const content =
    message.content ??
    (isRecord(message.kwargs) ? message.kwargs.content : undefined) ??
    (isRecord(message.data) ? message.data.content : undefined);
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return undefined;
  }
  const text = content
    .filter(isRecord)
    .map((part) => (typeof part.text === 'string' ? part.text : ''))
    .join('');
  return text === '' ? undefined : text;
}

function findLastMessageText(value: unknown, role: string): string | undefined {
  const messages = getMessageArray(value);
  if (messages == null) {
    return undefined;
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    if (getMessageRole(messages[i]) !== role) {
      continue;
    }
    const text = getMessageText(messages[i]);
    if (text != null && text.trim() !== '') {
      return text;
    }
  }
  return undefined;
}

/**
 * A summarize-only run answers with its summary, carried in state as
 * `manualSummary`, rather than with an assistant message.
 */
function getManualSummary(value: unknown): string | undefined {
  if (!isRecord(value) || typeof value.manualSummary !== 'string') {
    return undefined;
  }
  const text = value.manualSummary.trim();
  return text === '' ? undefined : text;
}

function normalizeToolCall(value: unknown): SerializedToolCall | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const fn = value.function;
  if (isRecord(fn) && typeof fn.name === 'string') {
    return {
      name: fn.name,
      args: parseAttributeValue(fn.arguments),
      ...(typeof value.id === 'string' ? { id: value.id } : {}),
    };
  }
  if (typeof value.name !== 'string') {
    return undefined;
  }
  return {
    name: value.name,
    args: value.args ?? {},
    ...(typeof value.id === 'string' ? { id: value.id } : {}),
  };
}

function getMessageToolCalls(
  message: Record<string, unknown>
): SerializedToolCall[] {
  const rawCalls =
    message.tool_calls ??
    (isRecord(message.kwargs) ? message.kwargs.tool_calls : undefined) ??
    (isRecord(message.additional_kwargs)
      ? message.additional_kwargs.tool_calls
      : undefined) ??
    (isRecord(message.data) ? message.data.tool_calls : undefined);
  if (!Array.isArray(rawCalls)) {
    return [];
  }
  const calls: SerializedToolCall[] = [];
  for (const rawCall of rawCalls) {
    const call = normalizeToolCall(rawCall);
    if (call != null) {
      calls.push(call);
    }
  }
  return calls;
}

/**
 * Id-bearing `invalid_tool_calls` on the assistant turn. ToolNode pairs these
 * with synthesized error results (and an invalid-only turn is routed on them
 * alone), so the tool-dispatch span must count them as part of the executing
 * batch — otherwise an invalid-only dispatch finds zero calls and the span
 * keeps the full serialized graph state as its input, and a mixed dispatch
 * silently omits the malformed call. `args` stays the raw unparsed string.
 */
/** Tool-result ids present in the serialized state — ToolNode's
 *  `!toolMessageIds.has(id)` execution filter, mirrored for the span. */
function getToolResultIds(messages: Record<string, unknown>[]): Set<string> {
  const ids = new Set<string>();
  for (const message of messages) {
    if (getMessageRole(message) !== 'tool') {
      continue;
    }
    const rawId =
      message.tool_call_id ??
      (isRecord(message.kwargs) ? message.kwargs.tool_call_id : undefined) ??
      (isRecord(message.data) ? message.data.tool_call_id : undefined);
    if (typeof rawId === 'string' && rawId !== '') {
      ids.add(rawId);
    }
  }
  return ids;
}

function getMessageInvalidToolCalls(
  message: Record<string, unknown>,
  answeredIds: ReadonlySet<string>
): SerializedToolCall[] {
  const rawCalls =
    message.invalid_tool_calls ??
    (isRecord(message.kwargs)
      ? message.kwargs.invalid_tool_calls
      : undefined) ??
    (isRecord(message.data) ? message.data.invalid_tool_calls : undefined);
  if (!Array.isArray(rawCalls)) {
    return [];
  }
  const calls: SerializedToolCall[] = [];
  for (const rawCall of rawCalls) {
    // Same attribution predicate ToolNode executes with (id-bearing,
    // non-server) so the span never claims calls the node deliberately skips.
    if (
      !isRecord(rawCall) ||
      typeof rawCall.id !== 'string' ||
      rawCall.id === '' ||
      answeredIds.has(rawCall.id) ||
      rawCall.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
    ) {
      continue;
    }
    // Same name fallback ToolNode synthesizes with — a nameless attributable
    // call still gets a result, so it must still count in the span input.
    const call = normalizeToolCall(
      typeof rawCall.name === 'string' && rawCall.name !== ''
        ? rawCall
        : { ...rawCall, name: 'unknown' }
    );
    if (call != null) {
      calls.push(call);
    }
  }
  return calls;
}

/** The serialized message's own id (uuid), NOT the LC-serialization type id
 *  array that `message.id` carries in constructor dumps. */
function getSerializedMessageId(
  message: Record<string, unknown>
): string | undefined {
  const kwargsId = isRecord(message.kwargs) ? message.kwargs.id : undefined;
  const dataId = isRecord(message.data) ? message.data.id : undefined;
  const rawId = message.id;
  const id = kwargsId ?? dataId ?? rawId;
  return typeof id === 'string' && id !== '' ? id : undefined;
}

/** Latest assistant turn's tool calls — the calls this tool node is executing. */
function findPendingToolCalls(value: unknown): SerializedToolCall[] {
  const messages = getMessageArray(value);
  if (messages == null) {
    return [];
  }
  /**
   * Invalid calls count only where ToolNode's own gate lets them execute:
   * the messages-state form (a bare-array state means the node returns a
   * plain output list and skips invalid handling) with an id-bearing
   * assistant message (no id, no reducer upsert). Mirrors
   * `canPromoteInvalidCalls` so the span never reports skipped calls.
   */
  const invalidCallsApply = !Array.isArray(value);
  const answeredIds = invalidCallsApply
    ? getToolResultIds(messages)
    : undefined;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (getMessageRole(messages[i]) !== 'assistant') {
      continue;
    }
    const calls = [
      ...getMessageToolCalls(messages[i]),
      ...(invalidCallsApply && getSerializedMessageId(messages[i]) != null
        ? getMessageInvalidToolCalls(messages[i], answeredIds!)
        : []),
    ];
    if (calls.length > 0) {
      return calls;
    }
  }
  return [];
}

function getRootSpanParentId(span: ReadableSpan): string | undefined {
  const legacyParent = (span as { parentSpanId?: string }).parentSpanId;
  if (typeof legacyParent === 'string' && legacyParent !== '') {
    return legacyParent;
  }
  const parentContext = (span as { parentSpanContext?: { spanId?: string } })
    .parentSpanContext;
  const spanId = parentContext?.spanId;
  return typeof spanId === 'string' && spanId !== '' ? spanId : undefined;
}

function isRootSpan(span: ReadableSpan): boolean {
  return getRootSpanParentId(span) == null;
}

/**
 * LangGraph plumbing observations that add noise without information:
 * the duplicated `__start__` channel-seed nodes and anonymous
 * `RunnableLambda` pass-throughs (Langfuse team feedback items 4 & 5).
 * Internal ToolNode batch spans are disabled at their source so traced child
 * tools retain an exported parent. Explicitly traced ToolNodes are preserved.
 */
export function shouldDropLangfuseSpan(spanName: string): boolean {
  return (
    spanName === LANGGRAPH_START_NODE || spanName === ANONYMOUS_LAMBDA_NAME
  );
}

function shapeToolNodeSpan(span: MutableSpan): void {
  const inputKey = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
  span.name = TOOL_DISPATCH_SPAN_NAME;
  span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
    CHAIN_OBSERVATION_TYPE;
  const calls = findPendingToolCalls(
    parseAttributeValue(span.attributes[inputKey])
  );
  if (calls.length === 0) {
    return;
  }
  span.attributes[inputKey] = JSON.stringify(
    calls.map(({ name, args }) => ({ name, args }))
  );
}

function shapeAgentNodeSpan(span: MutableSpan): void {
  span.name = AGENT_NODE_SPAN_NAME;
  span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
    ROOT_OBSERVATION_TYPE;
}

function shapeGraphSpan(span: MutableSpan): void {
  if (span.name === LANGGRAPH_RUN_NAME) {
    span.name = STANDARD_GRAPH_RUN_NAME;
  }
  span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
    ROOT_OBSERVATION_TYPE;
}

function isAgentModelCallSpan(span: MutableSpan): boolean {
  if (
    span.name !== RUNNABLE_SEQUENCE_NAME &&
    span.name !== AGENT_MODEL_CALL_RUN_NAME
  ) {
    return false;
  }
  const node = getObservationMetadataAttribute(
    span.attributes,
    OBSERVATION_METADATA_LANGGRAPH_NODE
  );
  return (
    typeof node === 'string' && node.startsWith(LANGGRAPH_AGENT_NODE_PREFIX)
  );
}

function shapeAgentModelCallSpan(span: MutableSpan): void {
  span.name = AGENT_MODEL_CALL_RUN_NAME;
  span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
    CHAIN_OBSERVATION_TYPE;
}

function shapeActivityPhaseSpan(span: MutableSpan): void {
  span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
    CHAIN_OBSERVATION_TYPE;
}

function shapeGenerationSpan(span: MutableSpan): void {
  if (
    hasTraceTag(span, ACTIVITY_LABEL_TRACE_TAG) &&
    getObservationMetadataAttribute(
      span.attributes,
      OBSERVATION_METADATA_OPERATION
    ) === ACTIVITY_LABEL_RUN_NAME
  ) {
    span.name = ACTIVITY_LABEL_RUN_NAME;
    return;
  }
  if (
    hasTraceTag(span, REASONING_LABEL_TRACE_TAG) &&
    getObservationMetadataAttribute(
      span.attributes,
      OBSERVATION_METADATA_OPERATION
    ) === REASONING_LABEL_RUN_NAME
  ) {
    span.name = REASONING_LABEL_RUN_NAME;
    return;
  }
  if (
    hasTraceTag(span, ACTIVITY_PHASE_TRACE_TAG) &&
    getObservationMetadataAttribute(
      span.attributes,
      OBSERVATION_METADATA_OPERATION
    ) === ACTIVITY_PHASE_LABEL_RUN_NAME
  ) {
    span.name = ACTIVITY_PHASE_LABEL_RUN_NAME;
    return;
  }
  span.name = GENERATION_SPAN_NAME;
}

function isGraphSpan(span: MutableSpan): boolean {
  if (
    span.name !== LANGGRAPH_RUN_NAME &&
    span.name !== STANDARD_GRAPH_RUN_NAME &&
    span.name !== MULTI_AGENT_GRAPH_RUN_NAME
  ) {
    return false;
  }
  return (
    isRootSpan(span) ||
    typeof getObservationMetadataAttribute(
      span.attributes,
      OBSERVATION_METADATA_LANGGRAPH_NODE
    ) === 'string' ||
    hasTraceTag(span, AGENT_TRACE_TAG)
  );
}

function shapeConversationPayload(span: MutableSpan): void {
  const inputKey = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
  const outputKey = LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT;
  const question = findLastMessageText(
    parseAttributeValue(span.attributes[inputKey]),
    'user'
  );
  const output = parseAttributeValue(span.attributes[outputKey]);
  /** A summarize-only run's summary is its answer; any assistant message
   *  still in state is an older reply the run retained, not its output. */
  const answer =
    getManualSummary(output) ?? findLastMessageText(output, 'assistant');
  /** A generation that IS the trace root — a bare `model.invoke` with no
   *  wrapping chain, i.e. the activity-label path — is also the only
   *  observation carrying its own prompt: reducing its observation input
   *  would discard the SystemMessage from the one place it is traced.
   *  Chain/agent roots keep the full reduction; their child generations
   *  still record the complete prompt. */
  if (!isGenerationSpan(span)) {
    if (question != null) {
      span.attributes[inputKey] = question;
    }
    if (answer != null) {
      span.attributes[outputKey] = answer;
    }
  }
}

function isGenerationSpan(span: MutableSpan): boolean {
  const type = span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE];
  return typeof type === 'string' && type.toLowerCase() === 'generation';
}

function isToolSpan(span: MutableSpan): boolean {
  const type = span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE];
  return (
    typeof type === 'string' && type.toLowerCase() === TOOL_OBSERVATION_TYPE
  );
}

/**
 * Whether the span is a LangGraph node execution whose node id is the span
 * name — the shape of the outer workflow-agent node.
 */
function isWorkflowNodeSpan(span: MutableSpan): boolean {
  return (
    getObservationMetadataAttribute(
      span.attributes,
      OBSERVATION_METADATA_LANGGRAPH_NODE
    ) === span.name
  );
}

/**
 * LibreChat ephemeral agents are identified as
 * `endpoint__model___sender[____index]` (`__` encodes `:`; see LibreChat's
 * `encodeEphemeralAgentId`), and the outer workflow node carries that id as
 * its span name — so switching models renames the span (item 1). Returns the
 * stable human `sender` segment only when the name matches that encoding:
 * the `endpoint__model` prefix must contain both segments and — unlike
 * display names such as `LibreChat Agent: Ops___EU` — never contains
 * whitespace, so legitimate names that merely embed `___` are left alone.
 */
function extractEphemeralAgentSender(name: string): string | undefined {
  let workingId = name;
  const indexSeparator = workingId.lastIndexOf(EPHEMERAL_AGENT_INDEX_SEPARATOR);
  if (
    indexSeparator !== -1 &&
    /^\d+$/.test(
      workingId.slice(indexSeparator + EPHEMERAL_AGENT_INDEX_SEPARATOR.length)
    )
  ) {
    workingId = workingId.slice(0, indexSeparator);
  }
  const senderSeparator = workingId.indexOf(EPHEMERAL_AGENT_SENDER_SEPARATOR);
  if (senderSeparator <= 0) {
    return undefined;
  }
  const encodedPrefix = workingId.slice(0, senderSeparator);
  if (/\s/.test(encodedPrefix)) {
    return undefined;
  }
  const prefixParts = encodedPrefix.split('__');
  if (prefixParts.length < 2 || prefixParts.some((part) => part === '')) {
    return undefined;
  }
  const sender = workingId
    .slice(senderSeparator + EPHEMERAL_AGENT_SENDER_SEPARATOR.length)
    .replace(/__/g, ':');
  return sender === '' ? undefined : sender;
}

/** Workflow-agent node spans are always children of the run's root chain and
 *  always carry `langgraph_node` metadata equal to their name, so host-named
 *  run roots (`LibreChat Agent: <name>`) and ordinary chains that merely look
 *  like encoded ids (`pipeline__stage___EU`) are never rename candidates.
 *  Successful decodes become `agent` observations, matching the inner
 *  `agent=<id>` node shaping. */
function shapeEphemeralAgentNodeSpan(span: MutableSpan): void {
  if (isToolSpan(span) || isRootSpan(span) || !isWorkflowNodeSpan(span)) {
    return;
  }
  const sender = extractEphemeralAgentSender(span.name);
  if (sender == null) {
    return;
  }
  span.name = sender;
  span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
    ROOT_OBSERVATION_TYPE;
}

function hasTraceTag(span: MutableSpan, expectedTag: string): boolean {
  const tags = parseAttributeValue(
    span.attributes[LangfuseOtelSpanAttributes.TRACE_TAGS]
  );
  return (
    Array.isArray(tags) &&
    tags.some((tag) => typeof tag === 'string' && tag === expectedTag)
  );
}

function shapeRootObservationType(span: MutableSpan): void {
  if (isGenerationSpan(span)) {
    return;
  }
  if (
    hasTraceTag(span, TITLE_TRACE_TAG) ||
    (span.name === ACTIVITY_PHASE_RUN_NAME &&
      hasTraceTag(span, ACTIVITY_PHASE_TRACE_TAG))
  ) {
    span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
      CHAIN_OBSERVATION_TYPE;
    return;
  }
  if (hasTraceTag(span, AGENT_TRACE_TAG)) {
    span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
      ROOT_OBSERVATION_TYPE;
  }
}

/**
 * Reshapes spans per Langfuse-team feedback before export:
 * - `agent=<id>` / `tools=<id>` node names carry the ephemeral agent id
 *   (`provider__model`) — strip it so switching models doesn't break
 *   name-based logic (item 1). The outer workflow node is named with the
 *   bare agent id; ephemeral ids reduce to their stable sender name.
 * - Graph and prompt-to-model framework names become stable SDK operations:
 *   `AgentGraph` / `MultiAgentGraph` and `AgentModelCall`.
 * - LLM generation spans keep the provider client class name (`ChatOpenAI`,
 *   `AzureChatOpenAI`, …); rename ordinary calls to a provider-agnostic `llm`
 *   and label calls by their activity/reasoning role. The model stays on the
 *   generation's model attribute.
 * - Agent nodes become `agent` observations, while tool-dispatch nodes become
 *   stable `chain` observations whose input is scoped to the pending calls.
 *   Individual child calls remain `tool` observations (items 3 & 4).
 * - Agent trace roots become `agent` observations, while title and activity
 *   summary operations become `chain` observations. Root-observation
 *   input/output are reduced to the user question and assistant response when
 *   chat messages are available (item 2).
 */
export function shapeLangfuseSpan(span: ReadableSpan): void {
  const mutable = span as MutableSpan;
  delete mutable.attributes[DEPRECATED_TRACE_INPUT_ATTRIBUTE];
  delete mutable.attributes[DEPRECATED_TRACE_OUTPUT_ATTRIBUTE];
  redactCompactionSemanticIndexInput(mutable);
  const isGraphObservation = isGraphSpan(mutable);
  if (mutable.name.startsWith(LANGGRAPH_AGENT_NODE_PREFIX)) {
    shapeAgentNodeSpan(mutable);
  } else if (mutable.name.startsWith(LANGGRAPH_TOOL_NODE_PREFIX)) {
    shapeToolNodeSpan(mutable);
  } else if (isGraphObservation) {
    shapeGraphSpan(mutable);
  } else if (isAgentModelCallSpan(mutable)) {
    shapeAgentModelCallSpan(mutable);
  } else if (mutable.name === ACTIVITY_PHASE_RUN_NAME) {
    shapeActivityPhaseSpan(mutable);
  } else if (isGenerationSpan(mutable)) {
    shapeGenerationSpan(mutable);
  } else {
    shapeEphemeralAgentNodeSpan(mutable);
  }
  if (!isRootSpan(span)) {
    if (isGraphObservation) {
      shapeConversationPayload(mutable);
    }
    return;
  }
  shapeRootObservationType(mutable);
  shapeConversationPayload(mutable);
}

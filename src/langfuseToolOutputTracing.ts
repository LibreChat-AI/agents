import { LangfuseSpanProcessor } from '@langfuse/otel';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Context } from '@opentelemetry/api';
import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type * as t from '@/types';
import {
  LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
  hasToolOutputTracingConfig,
  normalizeToolName,
  resolveLangfuseConfig,
  resolveToolOutputTracingConfig,
  resolveLangfuseContentRedactionText,
  resolveLangfusePrivacyConfig,
} from '@/langfuseConfig';
import {
  shapeLangfuseSpan,
  shouldDropLangfuseSpan,
} from '@/langfuseTraceShaping';
import { resolveToolOutputTracingConfigForSpan } from '@/langfuseRuntimeScope';
import { LANGFUSE_OPERATION_METADATA_KEY } from '@/langfuseOperation';

export { LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT, resolveLangfuseConfig };

export const LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY =
  'librechatLangfuseObservationMetadata';

const LANGGRAPH_TOOL_NODE_PREFIX = 'tools=';
const SERVER_TOOL_RESULT_PREFIX = '{"serverToolResult":';
const SERVER_TOOL_RESULT_REPLAY_MARKER_KEY = 'librechatResponsesReplay';

const CHAT_ROLES = new Set([
  'assistant',
  'developer',
  'human',
  'system',
  'user',
]);

type SpanWithAttributes = ReadableSpan & {
  attributes: Record<string, unknown>;
};

type RedactionResult = {
  value: unknown;
  changed: boolean;
};

type RedactionContext = {
  generatedImageData: Set<string>;
  generatedImageIds: Set<string>;
  toolNamesByCallId: Map<string, string>;
};

const TOOL_OUTPUT_FIELD_KEYS = ['content', 'artifact'];
const MAX_OBSERVATION_METADATA_FIELDS = 32;
const MAX_OBSERVATION_METADATA_STRING_LENGTH = 200;
const OBSERVATION_METADATA_KEY_PATTERN = /^[A-Za-z][A-Za-z0-9_.-]{0,63}$/;

type ResponsesReplayOutputDescriptor = {
  nestedOutputFields?: Readonly<Record<string, readonly string[]>>;
  outputFields: readonly string[];
  resolveToolNameFromCallId?: boolean;
  toolName?: string;
};

const RESPONSES_REPLAY_OUTPUT_DESCRIPTORS: Readonly<
  Record<string, ResponsesReplayOutputDescriptor>
> = {
  local_shell_call_output: {
    outputFields: ['output'],
    toolName: 'local_shell',
  },
  shell_call_output: { outputFields: ['output'], toolName: 'shell' },
  apply_patch_call_output: {
    outputFields: ['output'],
    toolName: 'apply_patch',
  },
  program_output: { outputFields: ['result'], toolName: 'program' },
  code_interpreter_call: {
    outputFields: ['outputs'],
    toolName: 'code_interpreter',
  },
  mcp_call: { outputFields: ['output', 'error'], toolName: 'mcp' },
  mcp_list_tools: {
    outputFields: ['tools', 'error'],
    toolName: 'mcp_list_tools',
  },
  image_generation_call: {
    outputFields: ['result'],
    toolName: 'image_generation',
  },
  file_search_call: {
    outputFields: ['results'],
    toolName: 'file_search',
  },
  web_search_call: {
    nestedOutputFields: { action: ['sources'] },
    outputFields: ['results'],
    toolName: 'web_search',
  },
  tool_search_output: {
    outputFields: ['tools'],
    toolName: 'tool_search',
  },
  function_call_output: {
    outputFields: ['output'],
    resolveToolNameFromCallId: true,
  },
  custom_tool_call_output: {
    outputFields: ['output'],
    resolveToolNameFromCallId: true,
  },
  computer_call_output: {
    outputFields: ['output'],
    toolName: 'computer_use',
  },
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isPresent(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

function shouldApplyToolOutputRedaction(
  config: ResolvedLangfuseToolOutputTracingConfig
): boolean {
  return config.enabled === false || config.redactedToolNames.size > 0;
}

function toolNameMatches(
  toolName: string | undefined,
  config: ResolvedLangfuseToolOutputTracingConfig
): boolean {
  if (!isPresent(toolName)) {
    return false;
  }

  const normalizedToolName = normalizeToolName(toolName);
  if (config.redactedToolNameMatchMode === 'partial') {
    for (const redactedToolName of config.redactedToolNames) {
      if (normalizedToolName.includes(redactedToolName)) {
        return true;
      }
    }
    return false;
  }

  return config.redactedToolNames.has(normalizedToolName);
}

/** Whether a tool's outputs are excluded from tracing (global disable or
 *  `redactedToolNames` match). Exported for the activity-label prompt
 *  builder, whose prompt becomes Langfuse generation input. */
export function shouldRedactTool(
  toolName: string | undefined,
  config: ResolvedLangfuseToolOutputTracingConfig
): boolean {
  return config.enabled === false || toolNameMatches(toolName, config);
}

function getStringField(
  value: Record<string, unknown>,
  key: string
): string | undefined {
  const field = value[key];
  return typeof field === 'string' ? field : undefined;
}

function getNestedStringField(
  value: Record<string, unknown>,
  objectKey: string,
  fieldKey: string
): string | undefined {
  const nested = value[objectKey];
  if (!isRecord(nested)) {
    return undefined;
  }
  return getStringField(nested, fieldKey);
}

function getSerializedToolCallId(
  value: Record<string, unknown>
): string | undefined {
  return (
    getStringField(value, 'tool_call_id') ??
    getStringField(value, 'toolCallId') ??
    getStringField(value, 'call_id') ??
    getNestedStringField(value, 'kwargs', 'tool_call_id') ??
    getNestedStringField(value, 'additional_kwargs', 'tool_call_id') ??
    getNestedStringField(value, 'data', 'tool_call_id') ??
    (typeof value.id === 'string' ? value.id : undefined)
  );
}

function getSerializedToolName(
  value: Record<string, unknown>,
  redactionContext?: RedactionContext
): string | undefined {
  const role = getStringField(value, 'role');
  const explicitName =
    getStringField(value, 'name') ??
    getStringField(value, 'tool_name') ??
    getNestedStringField(value, 'function', 'name') ??
    getNestedStringField(value, 'kwargs', 'name') ??
    getNestedStringField(value, 'additional_kwargs', 'name') ??
    getNestedStringField(value, 'data', 'name') ??
    (role != null && role.toLowerCase() !== 'tool' ? role : undefined);

  if (explicitName != null) {
    return explicitName;
  }

  const toolCallId = getSerializedToolCallId(value);
  return toolCallId != null
    ? redactionContext?.toolNamesByCallId.get(toolCallId)
    : undefined;
}

function hasToolMessageIdentity(value: Record<string, unknown>): boolean {
  const type = getStringField(value, 'type') ?? getStringField(value, '_type');
  if (type === 'tool' || type === 'tool_message') {
    return true;
  }

  const id = value.id;
  if (
    Array.isArray(id) &&
    id.some((part) => typeof part === 'string' && part.includes('ToolMessage'))
  ) {
    return true;
  }

  if (
    'tool_call_id' in value ||
    getNestedStringField(value, 'kwargs', 'tool_call_id') != null ||
    getNestedStringField(value, 'additional_kwargs', 'tool_call_id') != null
  ) {
    return true;
  }

  const role = getStringField(value, 'role');
  return (
    role != null &&
    !CHAT_ROLES.has(role.toLowerCase()) &&
    ('content' in value || isRecord(value.kwargs) || isRecord(value.data))
  );
}

function hasAssistantMessageIdentity(value: Record<string, unknown>): boolean {
  const role = getStringField(value, 'role');
  if (role?.toLowerCase() === 'assistant') {
    return true;
  }
  const type = getStringField(value, 'type') ?? getStringField(value, '_type');
  if (type === 'ai' || type === 'assistant') {
    return true;
  }
  const id = value.id;
  return (
    Array.isArray(id) &&
    id.some(
      (part) =>
        typeof part === 'string' &&
        (part === 'AIMessage' || part === 'AIMessageChunk')
    )
  );
}

function redactToolContentFields(
  value: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig
): Record<string, unknown> {
  const next = { ...value };

  for (const outputKey of TOOL_OUTPUT_FIELD_KEYS) {
    if (outputKey in next) {
      next[outputKey] = config.redactionText;
    }
  }

  for (const nestedKey of ['kwargs', 'data', 'additional_kwargs']) {
    const nested = next[nestedKey];
    if (!isRecord(nested)) {
      continue;
    }
    const nextNested = { ...nested };
    let changed = false;
    for (const outputKey of TOOL_OUTPUT_FIELD_KEYS) {
      if (outputKey in nextNested) {
        nextNested[outputKey] = config.redactionText;
        changed = true;
      }
    }
    if (changed) {
      next[nestedKey] = nextNested;
    }
  }

  return next;
}

function redactResponsesReplayOutput(
  value: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig,
  redactionContext: RedactionContext
): RedactionResult | undefined {
  const type = getStringField(value, 'type');
  if (
    type == null ||
    !Object.prototype.hasOwnProperty.call(
      RESPONSES_REPLAY_OUTPUT_DESCRIPTORS,
      type
    )
  ) {
    return undefined;
  }
  const descriptor =
    RESPONSES_REPLAY_OUTPUT_DESCRIPTORS[
      type as keyof typeof RESPONSES_REPLAY_OUTPUT_DESCRIPTORS
    ];
  const hasDirectOutput = descriptor.outputFields.some(
    (outputField) => outputField in value
  );
  const hasNestedOutput = Object.entries(
    descriptor.nestedOutputFields ?? {}
  ).some(([objectField, outputFields]) => {
    const nested = value[objectField];
    return (
      isRecord(nested) &&
      outputFields.some((outputField) => outputField in nested)
    );
  });
  if (!hasDirectOutput && !hasNestedOutput) {
    return undefined;
  }
  let toolName = descriptor.toolName;
  if (descriptor.resolveToolNameFromCallId === true) {
    toolName = getSerializedToolName(value, redactionContext);
  } else if (type === 'mcp_call') {
    toolName = getStringField(value, 'name') ?? descriptor.toolName;
  }
  if (!shouldRedactTool(toolName, config)) {
    return undefined;
  }
  const redacted = { ...value };
  for (const outputField of descriptor.outputFields) {
    if (outputField in redacted) {
      redacted[outputField] = config.redactionText;
    }
  }
  for (const [objectField, outputFields] of Object.entries(
    descriptor.nestedOutputFields ?? {}
  )) {
    const nested = redacted[objectField];
    if (!isRecord(nested)) {
      continue;
    }
    const redactedNested = { ...nested };
    for (const outputField of outputFields) {
      if (outputField in redactedNested) {
        redactedNested[outputField] = config.redactionText;
      }
    }
    redacted[objectField] = redactedNested;
  }
  return {
    value: redacted,
    changed: true,
  };
}

function redactStandardServerToolResult(
  value: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig,
  redactionContext: RedactionContext
): RedactionResult | undefined {
  if (
    getStringField(value, 'type') !== 'server_tool_call_result' ||
    !('output' in value)
  ) {
    return undefined;
  }
  const toolName =
    getNestedStringField(value, 'extras', 'name') ??
    getSerializedToolName(value, redactionContext);
  if (!shouldRedactTool(toolName, config)) {
    return undefined;
  }
  return {
    value: { ...value, output: config.redactionText },
    changed: true,
  };
}

function parseSerializedServerToolResultMarker(
  text: string
): { toolName?: string } | undefined {
  if (!text.startsWith(SERVER_TOOL_RESULT_PREFIX)) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(text) as unknown;
    if (!isRecord(parsed) || !isRecord(parsed.serverToolResult)) {
      return undefined;
    }
    const result = parsed.serverToolResult;
    if (result[SERVER_TOOL_RESULT_REPLAY_MARKER_KEY] !== true) {
      return undefined;
    }
    const status = getStringField(result, 'status');
    if (
      (status !== 'error' && status !== 'success') ||
      !Object.prototype.hasOwnProperty.call(result, 'output')
    ) {
      return undefined;
    }
    const toolName = getStringField(result, 'toolName');
    return toolName != null ? { toolName } : {};
  } catch {
    const match =
      /^\{"serverToolResult":\{"librechatResponsesReplay":true,(?:"toolName":("(?:\\.|[^"\\])*"),)?"status":"(?:error|success)","output":/.exec(
        text.slice(0, 512)
      );
    if (match == null) {
      return undefined;
    }
    const encodedToolName = match.at(1);
    if (encodedToolName == null) {
      return {};
    }
    try {
      const toolName = JSON.parse(encodedToolName) as unknown;
      return typeof toolName === 'string' ? { toolName } : {};
    } catch {
      return {};
    }
  }
}

function redactMarkedServerToolResult(
  value: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig,
  allowSerializedMarker: boolean
): RedactionResult | undefined {
  const type = getStringField(value, 'type');
  if (type !== 'text' && type !== 'image') {
    return undefined;
  }
  const text = type === 'text' ? getStringField(value, 'text') : undefined;
  const extras = value.extras;
  const marker = isRecord(extras)
    ? extras.librechatServerToolResult
    : undefined;
  const serializedMarker =
    allowSerializedMarker && text != null
      ? parseSerializedServerToolResultMarker(text)
      : undefined;
  if (!isRecord(marker) && serializedMarker == null) {
    return undefined;
  }
  const toolName = isRecord(marker)
    ? getStringField(marker, 'toolName')
    : serializedMarker?.toolName;
  if (!shouldRedactTool(toolName, config)) {
    return undefined;
  }
  if (type === 'image') {
    const redacted = { ...value };
    let changed = false;
    for (const outputField of ['data', 'url', 'fileId']) {
      if (outputField in redacted) {
        redacted[outputField] = config.redactionText;
        changed = true;
      }
    }
    return changed ? { value: redacted, changed: true } : undefined;
  }
  return {
    value: { ...value, text: config.redactionText },
    changed: true,
  };
}

function redactGeneratedImageBlock(
  value: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig,
  redactionContext: RedactionContext,
  isAssistantContent: boolean
): RedactionResult | undefined {
  if (
    getStringField(value, 'type') !== 'image' ||
    !shouldRedactTool('image_generation', config)
  ) {
    return undefined;
  }
  const extras = value.extras;
  const explicitMarker = isRecord(extras)
    ? extras.librechatServerToolResult
    : undefined;
  const explicitToolName = isRecord(explicitMarker)
    ? getStringField(explicitMarker, 'toolName')
    : undefined;
  if (
    explicitToolName != null &&
    normalizeToolName(explicitToolName) !== 'image_generation'
  ) {
    return undefined;
  }
  const id = getStringField(value, 'id');
  const data = getStringField(value, 'data');
  const metadata = value.metadata;
  const normalizedStatus = isRecord(metadata)
    ? getStringField(metadata, 'status')
    : undefined;
  const hasNormalizedGeneratedImageIdentity =
    isAssistantContent &&
    id?.startsWith('ig_') === true &&
    isPresent(normalizedStatus);
  const isGeneratedImage =
    explicitToolName != null ||
    hasNormalizedGeneratedImageIdentity ||
    (id != null
      ? redactionContext.generatedImageIds.has(id)
      : data != null && redactionContext.generatedImageData.has(data));
  if (!isGeneratedImage) {
    return undefined;
  }
  const redacted = { ...value };
  let changed = false;
  for (const outputField of ['data', 'url', 'fileId']) {
    if (outputField in redacted) {
      redacted[outputField] = config.redactionText;
      changed = true;
    }
  }
  return changed ? { value: redacted, changed: true } : undefined;
}

function collectRedactionContext(
  value: unknown,
  redactionContext: RedactionContext
): void {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectRedactionContext(item, redactionContext);
    }
    return;
  }

  if (!isRecord(value)) {
    return;
  }

  const toolCallId = getSerializedToolCallId(value);
  const toolName = getSerializedToolName(value);
  if (toolCallId != null && toolName != null) {
    redactionContext.toolNamesByCallId.set(toolCallId, toolName);
  }

  if (getStringField(value, 'type') === 'image_generation_call') {
    const id = getStringField(value, 'id');
    const result = getStringField(value, 'result');
    if (id != null) {
      redactionContext.generatedImageIds.add(id);
    }
    if (result != null) {
      redactionContext.generatedImageData.add(result);
    }
  }

  for (const child of Object.values(value)) {
    collectRedactionContext(child, redactionContext);
  }
}

function redactValue(
  value: unknown,
  config: ResolvedLangfuseToolOutputTracingConfig,
  redactionContext: RedactionContext,
  allowSerializedServerToolResult = false
): RedactionResult {
  if (Array.isArray(value)) {
    let changed = false;
    const next: unknown[] = [];
    for (const item of value) {
      const result = redactValue(
        item,
        config,
        redactionContext,
        allowSerializedServerToolResult
      );
      if (result.changed) {
        changed = true;
      }
      next.push(result.value);
    }
    return changed ? { value: next, changed } : { value, changed };
  }

  if (!isRecord(value)) {
    return { value, changed: false };
  }

  const allowSerializedMarker =
    allowSerializedServerToolResult || hasAssistantMessageIdentity(value);

  const replayOutput = redactResponsesReplayOutput(
    value,
    config,
    redactionContext
  );
  if (replayOutput != null) {
    return replayOutput;
  }
  const standardServerToolResult = redactStandardServerToolResult(
    value,
    config,
    redactionContext
  );
  if (standardServerToolResult != null) {
    return standardServerToolResult;
  }
  const markedServerToolResult = redactMarkedServerToolResult(
    value,
    config,
    allowSerializedMarker
  );
  if (markedServerToolResult != null) {
    return markedServerToolResult;
  }
  const generatedImageBlock = redactGeneratedImageBlock(
    value,
    config,
    redactionContext,
    allowSerializedMarker
  );
  if (generatedImageBlock != null) {
    return generatedImageBlock;
  }

  const toolName = getSerializedToolName(value, redactionContext);
  if (hasToolMessageIdentity(value) && shouldRedactTool(toolName, config)) {
    return {
      value: redactToolContentFields(value, config),
      changed: true,
    };
  }

  let changed = false;
  const next: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    const result = redactValue(
      child,
      config,
      redactionContext,
      allowSerializedMarker
    );
    if (result.changed) {
      changed = true;
    }
    next[key] = result.value;
  }

  return changed ? { value: next, changed } : { value, changed };
}

function redactSerializedValue(
  value: unknown,
  config: ResolvedLangfuseToolOutputTracingConfig
): RedactionResult {
  const redactionContext: RedactionContext = {
    generatedImageData: new Set(),
    generatedImageIds: new Set(),
    toolNamesByCallId: new Map(),
  };
  if (typeof value !== 'string') {
    collectRedactionContext(value, redactionContext);
    return redactValue(value, config, redactionContext);
  }

  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return { value, changed: false };
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    collectRedactionContext(parsed, redactionContext);
    const result = redactValue(parsed, config, redactionContext);
    return result.changed
      ? { value: JSON.stringify(result.value), changed: true }
      : { value, changed: false };
  } catch {
    // OpenTelemetry truncates string attributes before span processors run.
    // Langfuse serializes structured message values to JSON first, so a
    // truncated attribute can still contain tool output even though it is no
    // longer parseable. Fail closed for JSON-shaped attributes whenever tool
    // output redaction is active instead of exporting a partial secret.
    return { value: config.redactionText, changed: true };
  }
}

function redactAttribute(
  attributes: Record<string, unknown>,
  key: string,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  if (!(key in attributes)) {
    return;
  }

  const result = redactSerializedValue(attributes[key], config);
  if (result.changed) {
    attributes[key] = result.value;
  }
}

function isToolObservation(attributes: Record<string, unknown>): boolean {
  const type = attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE];
  return typeof type === 'string' && type.toLowerCase() === 'tool';
}

function classifyLangGraphToolNodeSpan(
  attributes: Record<string, unknown>
): void {
  const type = attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE];
  if (typeof type !== 'string' || type.toLowerCase() !== 'span') {
    return;
  }

  const langGraphNode =
    attributes[
      `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`
    ];
  if (
    typeof langGraphNode === 'string' &&
    langGraphNode.startsWith(LANGGRAPH_TOOL_NODE_PREFIX)
  ) {
    attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] = 'tool';
  }
}

export function classifyLangfuseToolNodeSpan(span: ReadableSpan): void {
  classifyLangGraphToolNodeSpan((span as SpanWithAttributes).attributes);
}

function parseSerializedValue(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return undefined;
  }
}

function getToolMessageArtifact(
  value: unknown
): Record<string, unknown> | undefined {
  const parsed = parseSerializedValue(value);
  if (!isRecord(parsed)) {
    return undefined;
  }
  if (isRecord(parsed.artifact)) {
    return parsed.artifact;
  }
  if (isRecord(parsed.kwargs) && isRecord(parsed.kwargs.artifact)) {
    return parsed.kwargs.artifact;
  }
  return undefined;
}

export function getToolObservationMetadata(
  artifact: unknown
): Record<string, string | number | boolean> {
  // This reserved artifact field is for trusted host adapters. Never copy
  // arbitrary tool or model output into it without an explicit allowlist.
  const metadata = isRecord(artifact)
    ? artifact[LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]
    : undefined;
  const result: Record<string, string | number | boolean> = {};
  if (!isRecord(metadata)) {
    return result;
  }

  let promoted = 0;
  for (const [key, value] of Object.entries(metadata)) {
    if (
      promoted >= MAX_OBSERVATION_METADATA_FIELDS ||
      !OBSERVATION_METADATA_KEY_PATTERN.test(key) ||
      !(
        typeof value === 'number' ||
        typeof value === 'boolean' ||
        (typeof value === 'string' &&
          value.length <= MAX_OBSERVATION_METADATA_STRING_LENGTH)
      )
    ) {
      continue;
    }
    if (typeof value === 'number' && !Number.isFinite(value)) {
      continue;
    }
    result[key] = value;
    promoted += 1;
  }
  return result;
}

function promoteToolObservationMetadata(
  span: ReadableSpan,
  attributes: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  if (!isToolObservation(attributes) || !shouldRedactTool(span.name, config)) {
    return;
  }
  const metadata = getToolObservationMetadata(
    getToolMessageArtifact(
      attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
    )
  );
  for (const [key, value] of Object.entries(metadata)) {
    const attributeKey = `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.${key}`;
    if (!Object.prototype.hasOwnProperty.call(attributes, attributeKey)) {
      attributes[attributeKey] = value;
    }
  }
}

function redactToolObservationOutput(
  span: ReadableSpan,
  attributes: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  if (
    !(
      isToolObservation(attributes) &&
      shouldRedactTool(span.name, config) &&
      LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT in attributes
    )
  ) {
    return;
  }

  attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] =
    config.redactionText;
}

export function redactLangfuseSpanToolOutputs(
  span: ReadableSpan,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  const attributes = (span as SpanWithAttributes).attributes;
  classifyLangfuseToolNodeSpan(span);

  if (!shouldApplyToolOutputRedaction(config)) {
    return;
  }

  promoteToolObservationMetadata(span, attributes, config);
  redactToolObservationOutput(span, attributes, config);

  for (const key of [
    LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
  ]) {
    redactAttribute(attributes, key, config);
  }
}

/**
 * Run-correlation identity preserved from masked metadata values. These keys
 * are SDK-written identifiers (never user content) that Langfuse consumers
 * need to link observations back to LibreChat runs, so `metricsOnly` keeps
 * them while every other metadata entry is dropped.
 */
const PRESERVED_TRACE_IDENTITY_KEYS = [
  'messageId',
  'parentMessageId',
  'agentId',
  'agentName',
  'sourceRunId',
  'responseId',
  LANGFUSE_OPERATION_METADATA_KEY,
] as const;

function preserveTraceIdentity(value: unknown, redactionText: string): string {
  if (typeof value !== 'string') {
    return redactionText;
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith('{')) {
    return redactionText;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return redactionText;
  }
  if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return redactionText;
  }
  const source = parsed as Record<string, unknown>;
  const preserved: Record<string, unknown> = {};
  for (const key of PRESERVED_TRACE_IDENTITY_KEYS) {
    if (key in source) {
      preserved[key] = source[key];
    }
  }
  return Object.keys(preserved).length > 0
    ? JSON.stringify(preserved)
    : redactionText;
}

/** Metadata also rides flattened, one attribute per entry. */
const PRESERVED_TRACE_IDENTITY_KEY_SET = new Set<string>(
  PRESERVED_TRACE_IDENTITY_KEYS
);

function redactFlattenedMetadata(attributes: Record<string, unknown>): void {
  const prefixes = [
    `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.`,
    `${LangfuseOtelSpanAttributes.TRACE_METADATA}.`,
  ];
  for (const key of Object.keys(attributes)) {
    const prefix = prefixes.find((candidate) => key.startsWith(candidate));
    if (prefix == null) {
      continue;
    }
    if (PRESERVED_TRACE_IDENTITY_KEY_SET.has(key.slice(prefix.length))) {
      continue;
    }
    delete attributes[key];
  }
}

/**
 * Applies the `metricsOnly` policy to a span's content-bearing attributes
 * before export.
 *
 * Redaction happens here rather than through the Langfuse span processor's
 * `mask` callback because the callback cannot tell which attribute it is
 * handling: preserving run-correlation identity in metadata would also
 * preserve same-named keys in user content (a tool returning
 * `{"messageId": "..."}`), and a blanket replacement would strip the
 * identity Langfuse consumers need. Per-attribute, input and output values
 * are replaced wholesale while metadata keeps only SDK-written identifiers;
 * flattened metadata entries are dropped unless they carry that identity.
 *
 * Model parameters are redacted too: they can embed content-bearing values
 * (caller-supplied stop strings, user identifiers, tool schemas and
 * descriptions). The model name, usage, and cost attributes stay.
 *
 * Fail closed: if any step throws, every masked family is replaced with the
 * redaction text instead of exporting the value verbatim.
 */
function redactLangfuseSpanContent(
  span: ReadableSpan,
  privacy: t.LangfusePrivacyConfig
): void {
  const redactionText = resolveLangfuseContentRedactionText(privacy);
  const attributes = span.attributes;
  const contentKeys = [
    LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
    LangfuseOtelSpanAttributes.TRACE_INPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
    LangfuseOtelSpanAttributes.TRACE_OUTPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS,
  ];
  const metadataKeys = [
    LangfuseOtelSpanAttributes.OBSERVATION_METADATA,
    LangfuseOtelSpanAttributes.TRACE_METADATA,
  ];
  try {
    for (const key of contentKeys) {
      if (key in attributes) {
        attributes[key] = redactionText;
      }
    }
    for (const key of metadataKeys) {
      if (key in attributes) {
        attributes[key] = preserveTraceIdentity(attributes[key], redactionText);
      }
    }
    redactFlattenedMetadata(attributes);
    redactLangfuseSpanStatusContent(span, redactionText);
  } catch {
    for (const key of [...contentKeys, ...metadataKeys]) {
      if (key in attributes) {
        attributes[key] = redactionText;
      }
    }
    try {
      redactFlattenedMetadata(attributes);
      redactLangfuseSpanStatusContent(span, redactionText);
    } catch {
      // Status redaction is best-effort; content attributes above are the
      // values that must never export verbatim.
    }
  }
}

/**
 * Redacts content that rides outside the masked attribute families: the
 * OTel status message and exception events. Error strings routinely embed
 * request data (a tool reporting an invalid user value, an upstream
 * response body). The status code stays: error vs. ok is operational data.
 */
function redactLangfuseSpanStatusContent(
  span: ReadableSpan,
  redactionText: string
): void {
  if (isPresent(span.status.message)) {
    span.status.message = redactionText;
  }
  const statusMessageKey =
    LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE;
  if (statusMessageKey in span.attributes) {
    span.attributes[statusMessageKey] = redactionText;
  }
  for (const event of span.events) {
    if (event.name !== 'exception') {
      continue;
    }
    for (const key of ['exception.message', 'exception.stacktrace']) {
      if (event.attributes != null && key in event.attributes) {
        event.attributes[key] = redactionText;
      }
    }
  }
}

export function prepareLangfuseSpanForExport(
  span: ReadableSpan,
  config?: ResolvedLangfuseToolOutputTracingConfig,
  privacy?: t.LangfusePrivacyConfig
): void {
  classifyLangfuseToolNodeSpan(span);
  if (config != null) {
    redactLangfuseSpanToolOutputs(span, config);
  }
  shapeLangfuseSpan(span);
  if (privacy?.mode === 'metricsOnly') {
    redactLangfuseSpanContent(span, privacy);
  }
}

class ToolOutputRedactingLangfuseSpanProcessor implements SpanProcessor {
  private readonly processor: LangfuseSpanProcessor;
  private readonly fallbackConfig?: ResolvedLangfuseToolOutputTracingConfig;
  private readonly privacy?: t.LangfusePrivacyConfig;
  private readonly spanConfigs = new WeakMap<
    object,
    ResolvedLangfuseToolOutputTracingConfig
  >();

  constructor(
    params?: LangfuseSpanProcessorParams,
    fallbackConfig?: ResolvedLangfuseToolOutputTracingConfig,
    privacy?: t.LangfusePrivacyConfig
  ) {
    this.processor = new LangfuseSpanProcessor(params);
    this.fallbackConfig = fallbackConfig;
    this.privacy = privacy;
  }

  onStart(span: Span, parentContext: Context): void {
    if (shouldDropLangfuseSpan(span.name)) {
      return;
    }
    const config =
      resolveToolOutputTracingConfigForSpan(parentContext) ??
      this.fallbackConfig;
    if (config != null) {
      this.spanConfigs.set(span, config);
    }
    this.processor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    if (shouldDropLangfuseSpan(span.name)) {
      return;
    }
    const config = this.spanConfigs.get(span) ?? this.fallbackConfig;
    prepareLangfuseSpanForExport(span, config, this.privacy);
    this.processor.onEnd(span);
  }

  forceFlush(): Promise<void> {
    return this.processor.forceFlush();
  }

  shutdown(): Promise<void> {
    return this.processor.shutdown();
  }
}

export function createLangfuseSpanProcessor(
  params?: LangfuseSpanProcessorParams,
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): SpanProcessor {
  const fallbackConfig = hasToolOutputTracingConfig(runLangfuse, agentLangfuse)
    ? resolveToolOutputTracingConfig(runLangfuse, agentLangfuse)
    : undefined;
  return new ToolOutputRedactingLangfuseSpanProcessor(
    params,
    fallbackConfig,
    resolveLangfusePrivacyConfig(runLangfuse)
  );
}

function hasLangfuseEnvKeys(): boolean {
  return (
    isPresent(process.env.LANGFUSE_SECRET_KEY) &&
    isPresent(process.env.LANGFUSE_PUBLIC_KEY)
  );
}

function hasLangfuseConfigKeys(langfuse?: t.LangfuseConfig): boolean {
  if (langfuse == null) {
    return false;
  }
  return isPresent(langfuse.secretKey) && isPresent(langfuse.publicKey);
}

export function shouldTraceToolNodeForLangfuse({
  runLangfuse,
  agentLangfuse,
}: {
  runLangfuse?: t.LangfuseConfig;
  agentLangfuse?: t.LangfuseConfig;
}): boolean {
  const langfuse = resolveLangfuseConfig(runLangfuse, agentLangfuse);
  if (langfuse?.enabled === false) {
    return false;
  }

  const explicit = langfuse?.toolNodeTracing?.enabled;
  if (explicit !== true) {
    return false;
  }

  return hasLangfuseConfigKeys(langfuse) || hasLangfuseEnvKeys();
}

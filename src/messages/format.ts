/* eslint-disable @typescript-eslint/no-explicit-any */
import {
  AIMessage,
  AIMessageChunk,
  ToolMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type {
  MessageContent,
  MessageContentImageUrl,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  BedrockReasoningContentText,
  ExtendedMessageContent,
  GoogleReasoningContentText,
  MessageContentComplex,
  ReasoningContentText,
  SummaryContentBlock,
  SummaryCoverage,
  ThinkingContentText,
  ToolCallContent,
  ToolResultContent,
  ToolCallPart,
  TPayload,
  TMessage,
  ProviderName,
  CompactionSemanticIndex,
  CompactionSemanticIndexEntry,
  CompactionSemanticIndexSnapshot,
} from '@/types';
import type {
  OrderedToolHistoryProjection,
  ToolHistoryPreparation,
  ToolHistoryCallMirrors,
} from './toolHistoryProjection';
import type {
  ProviderMessageAttribution,
  ProviderMessageProvenancePart,
} from './provenance';
import type { ProviderToolCallIndex } from './toolResultTypes';
import {
  appendProviderToolCallDescriptor,
  consumeProviderToolResultPair,
  getBoundedProviderPairingArrayProperty,
  getProviderToolCallPartDescriptor,
  getProviderToolResultPartDescriptor,
  hasStructurallyValidAnthropicWebSearchResultContent,
  isProviderToolContentPart,
} from './toolResultTypes';
import {
  hasBijectiveProviderContentPartMapping,
  inspectProviderMessageProvenance,
  inspectProviderSourceMessageIds,
  setFreshProviderMessageProvenance,
  setInvalidProviderMessageProvenance,
  setProviderMessageProvenance,
} from './provenance';
import {
  snapshotCompactionSemanticIndex,
  getCompactionSemanticIndexProvidedEntryCount,
  setCompactionSemanticIndexProvidedEntryCount,
} from '@/summarization/semanticIndex';
import {
  compactToolContent,
  getToolContentCharLength,
  isAtomicToolContentBlock,
  serializeStructuredValueBounded,
} from '@/utils/toolContent';
import {
  createToolHistoryPreparation,
  isToolHistoryCallMirror,
  recordToolHistoryCallMirror,
} from './toolHistoryProjection';
import {
  Providers,
  ContentTypes,
  Constants,
  COMPACTION_SEMANTIC_INDEX_LIMITS,
} from '@/common';
import { normalizeAnthropicToolCallId } from '@/llm/anthropic/utils/message_inputs';
import { toLangChainContent, toLangChainMessageFields } from './langchain';
import { flattenLegacyContent, isLegacyConvertible } from './content';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { isReasoningContentBlock } from './reasoningTypes';
import { emitAgentLog } from '@/utils/events';

interface MediaMessageParams {
  message: {
    role: string;
    content: string;
    name?: string;
    [key: string]: any;
  };
  mediaParts: MessageContentComplex[];
  endpoint?: Providers;
}

/**
 * Formats a message with media content (images, documents, videos, audios) to API payload format.
 *
 * @param params - The parameters for formatting.
 * @returns - The formatted message.
 */
export const formatMediaMessage = ({
  message,
  endpoint,
  mediaParts,
}: MediaMessageParams): {
  role: string;
  content: MessageContentComplex[];
  name?: string;
  [key: string]: any;
} => {
  // Create a new object to avoid mutating the input
  const result: {
    role: string;
    content: MessageContentComplex[];
    name?: string;
    [key: string]: any;
  } = {
    ...message,
    content: [] as MessageContentComplex[],
  };

  if (endpoint === Providers.ANTHROPIC) {
    result.content = [
      ...mediaParts,
      { type: ContentTypes.TEXT, text: message.content },
    ] as MessageContentComplex[];
    return result;
  }

  result.content = [
    { type: ContentTypes.TEXT, text: message.content },
    ...mediaParts,
  ] as MessageContentComplex[];

  return result;
};

interface MessageInput {
  role?: string;
  _name?: string;
  sender?: string;
  text?: string;
  content?: string | MessageContentComplex[];
  image_urls?: MessageContentImageUrl[];
  documents?: MessageContentComplex[];
  videos?: MessageContentComplex[];
  audios?: MessageContentComplex[];
  lc_id?: string[];
  [key: string]: any;
}

interface FormatMessageParams {
  message: MessageInput;
  userName?: string;
  assistantName?: string;
  endpoint?: Providers;
  langChain?: boolean;
}

export type LangChainMessageRole = 'system' | 'user' | 'assistant' | 'tool';

export type RoleBearingMessage<T extends BaseMessage = BaseMessage> = T & {
  role: LangChainMessageRole;
};

export function withMessageRole<T extends BaseMessage>(
  message: T,
  role: LangChainMessageRole
): RoleBearingMessage<T> {
  const roleMessage = message as T & { role?: LangChainMessageRole };
  if (roleMessage.role === role) {
    return roleMessage as RoleBearingMessage<T>;
  }
  Object.defineProperty(roleMessage, 'role', {
    value: role,
    writable: true,
    enumerable: false,
    configurable: true,
  });
  return roleMessage as RoleBearingMessage<T>;
}

interface FormattedMessage {
  role: string;
  content: string | MessageContentComplex[];
  name?: string;
  [key: string]: any;
}

/**
 * Formats a message to OpenAI payload format based on the provided options.
 *
 * @param params - The parameters for formatting.
 * @returns - The formatted message.
 */
export const formatMessage = ({
  message,
  userName,
  endpoint,
  assistantName,
  langChain = false,
}: FormatMessageParams):
  | FormattedMessage
  | RoleBearingMessage<HumanMessage>
  | RoleBearingMessage<AIMessage>
  | RoleBearingMessage<SystemMessage> => {
  // eslint-disable-next-line prefer-const
  let { role: _role, _name, sender, text, content: _content, lc_id } = message;
  if (lc_id && lc_id[2] && !langChain) {
    const roleMapping: Record<string, string> = {
      SystemMessage: 'system',
      HumanMessage: 'user',
      AIMessage: 'assistant',
    };
    _role = roleMapping[lc_id[2]] || _role;
  }
  const role =
    _role ??
    (sender != null && sender && sender.toLowerCase() === 'user'
      ? 'user'
      : 'assistant');
  const content = _content ?? text ?? '';
  const formattedMessage: FormattedMessage = {
    role,
    content,
  };

  // Set name fields first
  if (_name != null && _name) {
    formattedMessage.name = _name;
  }

  if (userName != null && userName && formattedMessage.role === 'user') {
    formattedMessage.name = userName;
  }

  if (
    assistantName != null &&
    assistantName &&
    formattedMessage.role === 'assistant'
  ) {
    formattedMessage.name = assistantName;
  }

  if (formattedMessage.name != null && formattedMessage.name) {
    // Conform to API regex: ^[a-zA-Z0-9_-]{1,64}$
    // https://community.openai.com/t/the-format-of-the-name-field-in-the-documentation-is-incorrect/175684/2
    formattedMessage.name = formattedMessage.name.replace(
      /[^a-zA-Z0-9_-]/g,
      '_'
    );

    if (formattedMessage.name.length > 64) {
      formattedMessage.name = formattedMessage.name.substring(0, 64);
    }
  }

  const { image_urls, documents, videos, audios } = message;
  const mediaParts: MessageContentComplex[] = [];

  if (Array.isArray(documents) && documents.length > 0) {
    for (const document of documents) {
      mediaParts.push(document);
    }
  }

  if (Array.isArray(videos) && videos.length > 0) {
    for (const video of videos) {
      mediaParts.push(video);
    }
  }

  if (Array.isArray(audios) && audios.length > 0) {
    for (const audio of audios) {
      mediaParts.push(audio);
    }
  }

  if (Array.isArray(image_urls) && image_urls.length > 0) {
    for (const imageUrl of image_urls) {
      mediaParts.push(imageUrl);
    }
  }

  if (mediaParts.length > 0 && role === 'user') {
    const mediaMessage = formatMediaMessage({
      message: {
        ...formattedMessage,
        content:
          typeof formattedMessage.content === 'string'
            ? formattedMessage.content
            : '',
      },
      mediaParts,
      endpoint,
    });

    if (!langChain) {
      return mediaMessage;
    }

    return withMessageRole(
      new HumanMessage(toLangChainMessageFields(mediaMessage)),
      'user'
    );
  }

  if (!langChain) {
    return formattedMessage;
  }

  if (role === 'user') {
    return withMessageRole(
      new HumanMessage(toLangChainMessageFields(formattedMessage)),
      'user'
    );
  } else if (role === 'assistant') {
    return withMessageRole(
      new AIMessage(toLangChainMessageFields(formattedMessage)),
      'assistant'
    );
  } else {
    return withMessageRole(
      new SystemMessage(toLangChainMessageFields(formattedMessage)),
      'system'
    );
  }
};

/**
 * Formats an array of messages for LangChain.
 *
 * @param messages - The array of messages to format.
 * @param formatOptions - The options for formatting each message.
 * @returns - The array of formatted LangChain messages.
 */
export const formatLangChainMessages = (
  messages: Array<MessageInput>,
  formatOptions: Omit<FormatMessageParams, 'message' | 'langChain'>
): Array<
  | RoleBearingMessage<HumanMessage>
  | RoleBearingMessage<AIMessage>
  | RoleBearingMessage<SystemMessage>
> => {
  return messages.map((msg) => {
    const formatted = formatMessage({
      ...formatOptions,
      message: msg,
      langChain: true,
    });
    return formatted as
      | RoleBearingMessage<HumanMessage>
      | RoleBearingMessage<AIMessage>
      | RoleBearingMessage<SystemMessage>;
  });
};

interface LangChainMessage {
  lc_kwargs?: {
    additional_kwargs?: Record<string, any>;
    [key: string]: any;
  };
  kwargs?: {
    additional_kwargs?: Record<string, any>;
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Formats a LangChain message object by merging properties from `lc_kwargs` or `kwargs` and `additional_kwargs`.
 *
 * @param message - The message object to format.
 * @returns - The formatted LangChain message.
 */
export const formatFromLangChain = (
  message: LangChainMessage
): Record<string, any> => {
  const kwargs = message.lc_kwargs ?? message.kwargs ?? {};
  const { additional_kwargs = {}, ...message_kwargs } = kwargs;
  return {
    ...message_kwargs,
    ...additional_kwargs,
  };
};

interface FormatAssistantMessageOptions {
  compactionSemanticIndex?: DerivedCompactionSemanticIndexCollector;
  intentToolNames?: ReadonlySet<string>;
  preserveUnpairedServerToolUses?: boolean;
  preserveReasoningContent?: boolean;
  provider?: ProviderName;
  retainedSourceContentEnd?: number;
  sourceMessageId?: string;
  sourceContentPartOffset?: number;
  sourceContentPartIndices?: readonly SourceContentPartIndices[];
  toolSourceContentPartIndices?: ReadonlySet<number>;
}

type SourceContentPartIndices = number | readonly number[];

export interface FormatAgentMessagesOptions {
  provider?: ProviderName;
  /** Emit flattenable text content as the joined string the legacy-content
   *  projection would produce, so the per-request `formatContentStrings` pass
   *  finds nothing to convert and every history message keeps its identity —
   *  which is what lets exact-count reuse skip re-tokenizing it. Set this if
   *  and only if the run's provider uses legacy string content. */
  legacyContent?: boolean;
  /** Reconstruct hidden `reasoning_content` from `THINK` parts onto prior
   *  tool-call messages. Explicit opt-in for OpenAI-compatible endpoints that
   *  replay reasoning across turns; defaults to on for DeepSeek thinking-mode. */
  preserveReasoningContent?: boolean;
  /** Skill names already primed fresh this turn (manual/always-apply). Their
   *  historical `skill` tool_calls are not reconstructed into a HumanMessage,
   *  so the same SKILL.md body is not injected twice in one request. */
  skipSkillBodyNames?: Set<string>;
  /** Derive bounded compaction guidance during the formatter's existing
   *  persisted-content analysis. Tool intents are accepted only for names
   *  the host identifies as semantic-label fields; business `intent`
   *  parameters must remain ordinary tool input. */
  compactionSemanticIndex?: {
    /** Previously committed, serializable guidance to evolve with entries
     *  derived from this payload. The formatter snapshots and validates
     *  caller-owned data before scanning the new messages. */
    baseSnapshot?: CompactionSemanticIndexSnapshot;
    intentToolNames?: ReadonlySet<string>;
  };
}

function extractReasoningContent(
  part: MessageContentComplex | undefined | null
): string {
  if (part == null || typeof part !== 'object') {
    return '';
  }
  if (part.type === ContentTypes.THINK) {
    const think = (part as ReasoningContentText).think;
    return typeof think === 'string' ? think : '';
  }
  if (part.type === ContentTypes.THINKING) {
    const thinking = (part as ThinkingContentText).thinking;
    return typeof thinking === 'string' ? thinking : '';
  }
  if (part.type === 'thinking_delta') {
    const thinking = readFoldedDataProperty(part, 'thinking');
    return typeof thinking === 'string' ? thinking : '';
  }
  if (part.type === ContentTypes.REASONING) {
    const reasoning = (part as GoogleReasoningContentText).reasoning;
    return typeof reasoning === 'string' ? reasoning : '';
  }
  if (part.type === 'reasoning-delta') {
    const reasoning = readFoldedDataProperty(part, 'reasoning');
    return typeof reasoning === 'string' ? reasoning : '';
  }
  if (part.type === ContentTypes.REASONING_CONTENT) {
    const reasoningText = (part as BedrockReasoningContentText).reasoningText;
    return typeof reasoningText.text === 'string' ? reasoningText.text : '';
  }
  return '';
}

type ServerToolInput = Exclude<NonNullable<ToolCallPart['args']>, string>;

function parseServerToolInput(args: ToolCallPart['args']): ServerToolInput {
  if (typeof args === 'string') {
    try {
      const parsed = JSON.parse(args) as unknown;
      return parsed != null &&
        typeof parsed === 'object' &&
        !Array.isArray(parsed)
        ? (parsed as ServerToolInput)
        : {};
    } catch {
      return {};
    }
  }
  return args != null && typeof args === 'object' ? args : {};
}

function getTextContent(part: MessageContentComplex): string {
  const { text } = part as { text?: unknown };
  return typeof text === 'string' ? text : '';
}

function hasMeaningfulAssistantContent(part: MessageContentComplex): boolean {
  if (part.type === ContentTypes.TEXT) {
    return getTextContent(part).trim().length > 0;
  }
  if (
    part.type === ContentTypes.TOOL_CALL ||
    part.type === ContentTypes.ERROR ||
    part.type === ContentTypes.AGENT_UPDATE ||
    part.type === ContentTypes.SUMMARY ||
    part.type === ContentTypes.ACTIVITY_LABEL
  ) {
    return false;
  }
  if (isReasoningContentBlock(part)) {
    return extractReasoningContent(part).trim().length > 0;
  }
  return part.type != null && part.type !== '';
}

function getToolUseId(part: MessageContentComplex): string | undefined {
  if (!('tool_use_id' in part) || typeof part.tool_use_id !== 'string') {
    return undefined;
  }
  return part.tool_use_id;
}

type CompactionSemanticContentPart = MessageContentComplex & {
  activity_label?: string;
  activity_label_type?: string;
  activity_label_revision?: number;
  activity_start_index?: number;
  pending?: boolean;
  reasoning_label?: string;
  reasoning_label_revision?: number;
  reasoning_label_status?: string;
  reasoning_label_step_id?: string;
};

type OrderedCompactionSemanticIndexEntry = {
  entry: CompactionSemanticIndexEntry;
  identityHash: number;
  order: number;
  retentionCount: number;
};

type CompactionSemanticEntryRing = {
  entries: OrderedCompactionSemanticIndexEntry[];
  cursor: number;
};

type CompactionSemanticTypeCoverage = {
  head: OrderedCompactionSemanticIndexEntry[];
  tail: CompactionSemanticEntryRing;
};

type CompactionSemanticCoverageState = {
  head: OrderedCompactionSemanticIndexEntry[];
  tail: CompactionSemanticEntryRing;
  latestByIdentityHash: Map<number, OrderedCompactionSemanticIndexEntry[]>;
  typeCoverage: Map<
    CompactionSemanticIndexEntry['type'],
    CompactionSemanticTypeCoverage
  >;
};

type CompactionSemanticRevisionFloor = {
  entry: CompactionSemanticIndexEntry;
  order: number;
};

type DerivedCompactionSemanticIndexCollector = {
  entries: CompactionSemanticIndexEntry[];
  entryCount: number;
  baseEntryCount: number;
  omittedBaseEntryCount: number;
  baseRevisionFloors?: Map<number, CompactionSemanticRevisionFloor[]>;
  coverage?: CompactionSemanticCoverageState;
};

const DERIVED_SEMANTIC_HEAD_LIMIT =
  COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries / 4;
const DERIVED_SEMANTIC_TAIL_LIMIT =
  COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries / 2;
const DERIVED_SEMANTIC_TYPE_EDGE_LIMIT =
  COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries / 32;
const DERIVED_SEMANTIC_TYPE_HASH: Readonly<
  Record<CompactionSemanticIndexEntry['type'], number>
> = Object.freeze({
  tool_intent: 1,
  tool_outcome: 2,
  activity_phase: 3,
  reasoning_label: 4,
});

function createDerivedCompactionSemanticIndexCollector(
  baseSnapshot?: CompactionSemanticIndexSnapshot
): DerivedCompactionSemanticIndexCollector {
  const collector: DerivedCompactionSemanticIndexCollector = {
    entries: [],
    entryCount: 0,
    baseEntryCount: 0,
    omittedBaseEntryCount: 0,
  };
  let baseEntries: CompactionSemanticIndex | undefined;
  let serializedProvidedEntryCount: number | undefined;
  try {
    baseEntries = baseSnapshot?.entries;
    serializedProvidedEntryCount = baseSnapshot?.providedEntryCount;
  } catch {
    return collector;
  }
  const snapshot = snapshotCompactionSemanticIndex(baseEntries);
  if (snapshot == null) {
    return collector;
  }
  const snapshotProvidedEntryCount =
    getCompactionSemanticIndexProvidedEntryCount(snapshot);
  const validSerializedProvidedEntryCount =
    typeof serializedProvidedEntryCount === 'number' &&
    Number.isSafeInteger(serializedProvidedEntryCount) &&
    serializedProvidedEntryCount >= snapshotProvidedEntryCount;
  const providedEntryCount =
    validSerializedProvidedEntryCount && serializedProvidedEntryCount != null
      ? serializedProvidedEntryCount
      : snapshotProvidedEntryCount;
  collector.omittedBaseEntryCount = Math.max(
    0,
    providedEntryCount - snapshot.length
  );
  for (let index = 0; index < snapshot.length; index++) {
    appendDerivedCompactionSemanticEntry(collector, snapshot[index], true);
  }
  collector.baseEntryCount = collector.entryCount;
  return collector;
}

function createCompactionSemanticCoverageState(): CompactionSemanticCoverageState {
  return {
    head: [],
    tail: { entries: [], cursor: 0 },
    latestByIdentityHash: new Map(),
    typeCoverage: new Map(),
  };
}

function getDerivedCompactionSemanticLocalId(
  entry: CompactionSemanticIndexEntry
): string {
  if (entry.type === 'reasoning_label') {
    return entry.reasoningStepId.trim();
  }
  if (entry.type !== 'activity_phase') {
    return entry.toolCallId.trim();
  }
  return '';
}

function hashDerivedCompactionSemanticIdentityString(
  initialHash: number,
  value: string
): number {
  let hash = initialHash;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }
  return hash;
}

function hashDerivedCompactionSemanticIdentity(
  entry: CompactionSemanticIndexEntry
): number {
  let hash = 2_166_136_261 ^ DERIVED_SEMANTIC_TYPE_HASH[entry.type];
  hash = Math.imul(hash, 16_777_619);
  hash = hashDerivedCompactionSemanticIdentityString(
    hash,
    entry.sourceMessageId.trim()
  );
  hash ^= entry.sourceContentIndex;
  hash = Math.imul(hash, 16_777_619);
  hash = hashDerivedCompactionSemanticIdentityString(
    hash,
    getDerivedCompactionSemanticLocalId(entry)
  );
  return hash >>> 0;
}

function entriesShareDerivedCompactionSemanticIdentity(
  left: CompactionSemanticIndexEntry,
  right: CompactionSemanticIndexEntry
): boolean {
  return (
    left.type === right.type &&
    left.sourceMessageId.trim() === right.sourceMessageId.trim() &&
    left.sourceContentIndex === right.sourceContentIndex &&
    getDerivedCompactionSemanticLocalId(left) ===
      getDerivedCompactionSemanticLocalId(right)
  );
}

function entriesHaveConflictingSemanticState(
  left: CompactionSemanticIndexEntry,
  right: CompactionSemanticIndexEntry
): boolean {
  return (
    left.status !== right.status ||
    (left.redacted === true) !== (right.redacted === true) ||
    left.text.replace(/\s+/g, ' ').trim() !==
      right.text.replace(/\s+/g, ' ').trim()
  );
}

function findCompactionSemanticRevisionFloor(
  collector: DerivedCompactionSemanticIndexCollector,
  entry: CompactionSemanticIndexEntry,
  identityHash: number
): CompactionSemanticRevisionFloor | undefined {
  return collector.baseRevisionFloors
    ?.get(identityHash)
    ?.find((floor) =>
      entriesShareDerivedCompactionSemanticIdentity(floor.entry, entry)
    );
}

function seedCompactionSemanticRevisionFloor(
  collector: DerivedCompactionSemanticIndexCollector,
  entry: CompactionSemanticIndexEntry,
  order: number
): void {
  collector.baseRevisionFloors ??= new Map();
  const identityHash = hashDerivedCompactionSemanticIdentity(entry);
  const existing = findCompactionSemanticRevisionFloor(
    collector,
    entry,
    identityHash
  );
  if (existing != null) {
    if (entry.revision > existing.entry.revision) {
      existing.entry = entry;
      existing.order = order;
    }
    return;
  }
  const floor = { entry, order };
  const bucket = collector.baseRevisionFloors.get(identityHash);
  if (bucket == null) {
    collector.baseRevisionFloors.set(identityHash, [floor]);
    return;
  }
  bucket.push(floor);
}

function updateCompactionSemanticRevisionFloor(
  collector: DerivedCompactionSemanticIndexCollector,
  orderedEntry: OrderedCompactionSemanticIndexEntry
): void {
  const floor = findCompactionSemanticRevisionFloor(
    collector,
    orderedEntry.entry,
    orderedEntry.identityHash
  );
  if (floor == null || orderedEntry.entry.revision <= floor.entry.revision) {
    return;
  }
  floor.entry = orderedEntry.entry;
  floor.order = orderedEntry.order;
}

function shouldRejectCompactionSemanticEntryBelowBaseFloor(
  collector: DerivedCompactionSemanticIndexCollector,
  orderedEntry: OrderedCompactionSemanticIndexEntry
): boolean {
  const floor = findCompactionSemanticRevisionFloor(
    collector,
    orderedEntry.entry,
    orderedEntry.identityHash
  );
  if (floor == null || orderedEntry.order === floor.order) {
    return false;
  }
  const currentBucket = collector.coverage?.latestByIdentityHash.get(
    orderedEntry.identityHash
  );
  const current = currentBucket?.find(({ entry }) =>
    entriesShareDerivedCompactionSemanticIdentity(entry, orderedEntry.entry)
  );
  if (current != null) {
    return false;
  }
  return orderedEntry.entry.revision <= floor.entry.revision;
}

function retainCompactionSemanticEntry(
  entry: OrderedCompactionSemanticIndexEntry
): void {
  entry.retentionCount++;
}

function releaseCompactionSemanticEntry(
  coverage: CompactionSemanticCoverageState,
  entry: OrderedCompactionSemanticIndexEntry
): void {
  entry.retentionCount--;
  if (
    entry.retentionCount === 0 &&
    coverage.latestByIdentityHash.has(entry.identityHash)
  ) {
    const bucket = coverage.latestByIdentityHash.get(entry.identityHash);
    if (bucket == null) {
      return;
    }
    const entryIndex = bucket.indexOf(entry);
    if (entryIndex >= 0) {
      bucket.splice(entryIndex, 1);
    }
    if (bucket.length === 0) {
      coverage.latestByIdentityHash.delete(entry.identityHash);
    }
  }
}

function appendCompactionSemanticEntryRing(
  coverage: CompactionSemanticCoverageState,
  ring: CompactionSemanticEntryRing,
  entry: OrderedCompactionSemanticIndexEntry,
  limit: number
): void {
  if (ring.entries.length < limit) {
    ring.entries.push(entry);
    retainCompactionSemanticEntry(entry);
    return;
  }
  releaseCompactionSemanticEntry(coverage, ring.entries[ring.cursor]);
  ring.entries[ring.cursor] = entry;
  retainCompactionSemanticEntry(entry);
  ring.cursor = (ring.cursor + 1) % limit;
}

function renewCompactionSemanticEntryRing(
  coverage: CompactionSemanticCoverageState,
  ring: CompactionSemanticEntryRing,
  entry: OrderedCompactionSemanticIndexEntry,
  limit: number
): void {
  const existingIndex = ring.entries.indexOf(entry);
  if (existingIndex < 0) {
    appendCompactionSemanticEntryRing(coverage, ring, entry, limit);
    return;
  }
  if (ring.entries.length < limit) {
    ring.entries.splice(existingIndex, 1);
    ring.entries.push(entry);
    return;
  }
  const newestIndex =
    (ring.cursor + ring.entries.length - 1) % ring.entries.length;
  let currentIndex = existingIndex;
  for (
    let offset = 0;
    currentIndex !== newestIndex && offset < ring.entries.length;
    offset++
  ) {
    const nextIndex = (currentIndex + 1) % ring.entries.length;
    ring.entries[currentIndex] = ring.entries[nextIndex];
    currentIndex = nextIndex;
  }
  ring.entries[newestIndex] = entry;
}

function renewCoverageBalancedCompactionSemanticEntry(
  coverage: CompactionSemanticCoverageState,
  entry: OrderedCompactionSemanticIndexEntry
): void {
  renewCompactionSemanticEntryRing(
    coverage,
    coverage.tail,
    entry,
    DERIVED_SEMANTIC_TAIL_LIMIT
  );
  const typeCoverage = coverage.typeCoverage.get(entry.entry.type);
  if (typeCoverage == null) {
    return;
  }
  renewCompactionSemanticEntryRing(
    coverage,
    typeCoverage.tail,
    entry,
    DERIVED_SEMANTIC_TYPE_EDGE_LIMIT
  );
}

function appendDerivedCompactionSemanticEntry(
  collector: DerivedCompactionSemanticIndexCollector,
  entry: CompactionSemanticIndexEntry,
  baseEntry = false
): void {
  let boundedEntry = entry;
  if (entry.status === 'pending') {
    boundedEntry = { ...entry, text: '' };
  } else if (
    entry.text.length > COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputTextChars
  ) {
    boundedEntry = { ...entry, text: '', redacted: true };
  }
  const order = collector.entryCount;
  collector.entryCount++;
  if (
    collector.coverage == null &&
    collector.entries.length < COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries
  ) {
    collector.entries.push(boundedEntry);
    if (!baseEntry && collector.baseRevisionFloors != null) {
      updateCompactionSemanticRevisionFloor(collector, {
        entry: boundedEntry,
        identityHash: hashDerivedCompactionSemanticIdentity(boundedEntry),
        order,
        retentionCount: 0,
      });
    }
    return;
  }
  const orderedEntry = {
    entry: boundedEntry,
    identityHash: hashDerivedCompactionSemanticIdentity(boundedEntry),
    order,
    retentionCount: 0,
  };
  if (collector.coverage == null) {
    for (let order = 0; order < collector.baseEntryCount; order++) {
      seedCompactionSemanticRevisionFloor(
        collector,
        collector.entries[order],
        order
      );
    }
    collector.coverage = createCompactionSemanticCoverageState();
    for (let order = 0; order < collector.entries.length; order++) {
      const bufferedEntry = {
        entry: collector.entries[order],
        identityHash: hashDerivedCompactionSemanticIdentity(
          collector.entries[order]
        ),
        order,
        retentionCount: 0,
      };
      if (
        shouldRejectCompactionSemanticEntryBelowBaseFloor(
          collector,
          bufferedEntry
        )
      ) {
        continue;
      }
      appendCoverageBalancedCompactionSemanticEntry(
        collector.coverage,
        bufferedEntry
      );
      if (order >= collector.baseEntryCount) {
        updateCompactionSemanticRevisionFloor(collector, bufferedEntry);
      }
    }
    collector.entries = [];
  }
  if (
    !baseEntry &&
    shouldRejectCompactionSemanticEntryBelowBaseFloor(collector, orderedEntry)
  ) {
    return;
  }
  appendCoverageBalancedCompactionSemanticEntry(
    collector.coverage,
    orderedEntry
  );
  if (!baseEntry) {
    updateCompactionSemanticRevisionFloor(collector, orderedEntry);
  }
}

function appendCoverageBalancedCompactionSemanticEntry(
  coverage: CompactionSemanticCoverageState,
  orderedEntry: OrderedCompactionSemanticIndexEntry
): void {
  const identityHash = orderedEntry.identityHash;
  const identityBucket = coverage.latestByIdentityHash.get(identityHash);
  const current = identityBucket?.find(({ entry }) =>
    entriesShareDerivedCompactionSemanticIdentity(entry, orderedEntry.entry)
  );
  if (current != null) {
    if (orderedEntry.entry.revision < current.entry.revision) {
      return;
    }
    if (orderedEntry.entry.revision === current.entry.revision) {
      if (
        entriesHaveConflictingSemanticState(current.entry, orderedEntry.entry)
      ) {
        current.entry = { ...current.entry, text: '', redacted: true };
        current.order = orderedEntry.order;
        renewCoverageBalancedCompactionSemanticEntry(coverage, current);
      }
      return;
    }
    current.entry = orderedEntry.entry;
    current.order = orderedEntry.order;
    renewCoverageBalancedCompactionSemanticEntry(coverage, current);
    return;
  }
  if (identityBucket == null) {
    coverage.latestByIdentityHash.set(identityHash, [orderedEntry]);
  } else {
    identityBucket.push(orderedEntry);
  }
  if (coverage.head.length < DERIVED_SEMANTIC_HEAD_LIMIT) {
    coverage.head.push(orderedEntry);
    retainCompactionSemanticEntry(orderedEntry);
  }
  appendCompactionSemanticEntryRing(
    coverage,
    coverage.tail,
    orderedEntry,
    DERIVED_SEMANTIC_TAIL_LIMIT
  );
  let typeCoverage = coverage.typeCoverage.get(orderedEntry.entry.type);
  if (typeCoverage == null) {
    typeCoverage = { head: [], tail: { entries: [], cursor: 0 } };
    coverage.typeCoverage.set(orderedEntry.entry.type, typeCoverage);
  }
  if (typeCoverage.head.length < DERIVED_SEMANTIC_TYPE_EDGE_LIMIT) {
    typeCoverage.head.push(orderedEntry);
    retainCompactionSemanticEntry(orderedEntry);
  }
  appendCompactionSemanticEntryRing(
    coverage,
    typeCoverage.tail,
    orderedEntry,
    DERIVED_SEMANTIC_TYPE_EDGE_LIMIT
  );
}

function finalizeDerivedCompactionSemanticIndexSnapshot(
  collector: DerivedCompactionSemanticIndexCollector | undefined
): CompactionSemanticIndexSnapshot | undefined {
  if (
    collector == null ||
    (collector.entryCount === 0 && collector.omittedBaseEntryCount === 0)
  ) {
    return undefined;
  }
  const providedEntryCount = Math.min(
    Number.MAX_SAFE_INTEGER,
    collector.entryCount + collector.omittedBaseEntryCount
  );
  if (collector.coverage == null) {
    setCompactionSemanticIndexProvidedEntryCount(
      collector.entries,
      providedEntryCount
    );
    return { entries: collector.entries, providedEntryCount };
  }
  const retained = new Map<number, CompactionSemanticIndexEntry>();
  const retain = (
    entries: readonly OrderedCompactionSemanticIndexEntry[]
  ): void => {
    for (const { entry, order } of entries) {
      retained.set(order, entry);
    }
  };
  retain(collector.coverage.head);
  retain(collector.coverage.tail.entries);
  for (const typeCoverage of collector.coverage.typeCoverage.values()) {
    retain(typeCoverage.head);
    retain(typeCoverage.tail.entries);
  }
  const result = [...retained.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, entry]) => entry);
  setCompactionSemanticIndexProvidedEntryCount(result, providedEntryCount);
  return { entries: result, providedEntryCount };
}

function collectCompactionSemanticEntriesFromPart(
  collector: DerivedCompactionSemanticIndexCollector,
  part: CompactionSemanticContentPart,
  sourceMessageId: string,
  sourceContentIndex: number,
  retainedSourceContentStart: number,
  retainedSourceContentEnd: number,
  intentToolNames?: ReadonlySet<string>,
  parsedToolInput?: ToolCallPart['args']
): void {
  if (
    sourceMessageId.length > COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars
  ) {
    return;
  }

  if (part.type === ContentTypes.TOOL_CALL) {
    if (
      sourceContentIndex < 0 ||
      sourceContentIndex >
        COMPACTION_SEMANTIC_INDEX_LIMITS.maxSourceContentIndex
    ) {
      return;
    }
    const toolCall = part.tool_call;
    const toolCallId = toolCall?.id;
    if (
      typeof toolCallId !== 'string' ||
      toolCallId === '' ||
      toolCallId.length > COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars
    ) {
      return;
    }
    if (
      typeof toolCall.name === 'string' &&
      intentToolNames?.has(toolCall.name) === true
    ) {
      const intent =
        parsedToolInput != null &&
        typeof parsedToolInput === 'object' &&
        !Array.isArray(parsedToolInput)
          ? parsedToolInput.intent
          : undefined;
      if (typeof intent === 'string' && intent !== '') {
        appendDerivedCompactionSemanticEntry(collector, {
          type: 'tool_intent',
          sourceMessageId,
          sourceContentIndex,
          revision: 0,
          status: 'committed',
          text: intent,
          toolCallId,
        });
      }
    }
    if (typeof toolCall.outcome === 'string' && toolCall.outcome !== '') {
      appendDerivedCompactionSemanticEntry(collector, {
        type: 'tool_outcome',
        sourceMessageId,
        sourceContentIndex,
        revision: 0,
        status: 'committed',
        text: toolCall.outcome,
        toolCallId,
      });
    }
    return;
  }

  if (part.type === ContentTypes.THINK) {
    if (
      sourceContentIndex < 0 ||
      sourceContentIndex >
        COMPACTION_SEMANTIC_INDEX_LIMITS.maxSourceContentIndex
    ) {
      return;
    }
    const reasoningStepId = part.reasoning_label_step_id;
    const revision = part.reasoning_label_revision;
    const status = part.reasoning_label_status;
    const text = part.reasoning_label;
    if (
      typeof reasoningStepId !== 'string' ||
      reasoningStepId === '' ||
      reasoningStepId.length >
        COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars ||
      typeof revision !== 'number' ||
      !Number.isSafeInteger(revision) ||
      revision < 0
    ) {
      return;
    }
    const malformedState =
      (status !== 'complete' && status !== 'streaming') ||
      typeof text !== 'string';
    appendDerivedCompactionSemanticEntry(collector, {
      type: 'reasoning_label',
      sourceMessageId,
      sourceContentIndex,
      revision,
      status:
        status === 'streaming' || malformedState ? 'pending' : 'committed',
      text: typeof text === 'string' ? text : '',
      reasoningStepId,
      ...(malformedState ? { redacted: true } : {}),
    });
    return;
  }

  if (
    part.type !== ContentTypes.ACTIVITY_LABEL ||
    part.activity_label_type !== 'phase'
  ) {
    return;
  }
  const activitySourceContentIndex = part.activity_start_index;
  if (
    typeof activitySourceContentIndex !== 'number' ||
    !Number.isSafeInteger(activitySourceContentIndex) ||
    activitySourceContentIndex < retainedSourceContentStart ||
    activitySourceContentIndex >= retainedSourceContentEnd ||
    activitySourceContentIndex >
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxSourceContentIndex
  ) {
    return;
  }
  const revision = part.activity_label_revision;
  if (
    revision !== undefined &&
    (typeof revision !== 'number' ||
      !Number.isSafeInteger(revision) ||
      revision < 0)
  ) {
    return;
  }
  const malformedPending =
    part.pending !== undefined && typeof part.pending !== 'boolean';
  const malformedText = typeof part.activity_label !== 'string';
  const malformedState = malformedPending || malformedText;
  const text =
    typeof part.activity_label === 'string' ? part.activity_label : '';
  appendDerivedCompactionSemanticEntry(collector, {
    type: 'activity_phase',
    sourceMessageId,
    sourceContentIndex: activitySourceContentIndex,
    revision: revision ?? 0,
    status: part.pending === true || malformedPending ? 'pending' : 'committed',
    text,
    ...(malformedState ? { redacted: true } : {}),
  });
}

function collectTrustedToolResultSourceContentPartIndices(
  content: readonly unknown[] | undefined,
  sourceContentPartOffset: number
): Set<number> | undefined {
  if (content == null) {
    return undefined;
  }
  let calls: ProviderToolCallIndex | undefined;
  let trusted: Set<number> | undefined;
  let previousPart: MessageContentComplex | null | undefined;
  for (let index = 0; index < content.length; index++) {
    const part = content[index] as MessageContentComplex | null | undefined;
    if (part == null) {
      previousPart = part;
      continue;
    }
    const call = getProviderToolCallPartDescriptor(part);
    if (call != null) {
      calls ??= new Map();
      appendProviderToolCallDescriptor(calls, call);
    }
    const result = getProviderToolResultPartDescriptor(part);
    if (result != null) {
      calls ??= new Map();
      if (consumeProviderToolResultPair(result, calls, previousPart)) {
        trusted ??= new Set();
        trusted.add(sourceContentPartOffset + index);
      }
    }
    previousPart = part;
  }
  return trusted;
}

function sourceContentPartIndicesAreTrustedToolResult(
  sourceContentPartIndices: SourceContentPartIndices,
  trustedToolSourceContentPartIndices: ReadonlySet<number> | undefined
): boolean {
  if (typeof sourceContentPartIndices === 'number') {
    return (
      trustedToolSourceContentPartIndices?.has(sourceContentPartIndices) ===
      true
    );
  }
  if (sourceContentPartIndices.length === 0) {
    return false;
  }
  for (const sourceContentPartIndex of sourceContentPartIndices) {
    if (
      trustedToolSourceContentPartIndices?.has(sourceContentPartIndex) !== true
    ) {
      return false;
    }
  }
  return true;
}

function isTrustedServerToolResult(
  part: MessageContentComplex,
  sourceContentPartIndices: SourceContentPartIndices,
  trustedToolSourceContentPartIndices: ReadonlySet<number> | undefined
): boolean {
  const toolUseId = getToolUseId(part);
  return (
    toolUseId?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) === true &&
    sourceContentPartIndicesAreTrustedToolResult(
      sourceContentPartIndices,
      trustedToolSourceContentPartIndices
    )
  );
}

function isServerToolResultForWire(part: MessageContentComplex): boolean {
  const toolUseId = getToolUseId(part);
  return (
    toolUseId?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) === true &&
    hasStructurallyValidAnthropicWebSearchResultContent(part)
  );
}

function getToolCallId(part: MessageContentComplex): string | undefined {
  if (part.type !== ContentTypes.TOOL_CALL) {
    return undefined;
  }
  const id = part.tool_call?.id;
  return typeof id === 'string' && id !== '' ? id : undefined;
}

function hasToolCallOutput(part: MessageContentComplex): boolean {
  if (part.type !== ContentTypes.TOOL_CALL) {
    return false;
  }
  const output = part.tool_call?.output;
  return output != null && output !== '';
}

function formatToolCallOutput(
  output: ToolCallPart['output'] | undefined
): MessageContent {
  if (output == null) {
    return '';
  }
  return compactToolContent(output, HARD_MAX_TOOL_RESULT_CHARS).content;
}

/**
 * Content for the synthetic assistant turn that separates a trailing steer
 * from the next user turn. Non-empty by necessity — see the push site.
 */
const STEER_ANCHOR_PLACEHOLDER = '_';

/**
 * Stand-in text for a user message whose content reduces to nothing visible
 * (`userContentHasNoVisibleText` below): providers with strict payload
 * validation (Anthropic, Bedrock, Maritaca) reject such a message — and with
 * it the entire request — while lenient ones (OpenAI, Google, xAI) accept it
 * silently, which hides the issue until the model list changes.
 */
const EMPTY_USER_MESSAGE_PLACEHOLDER = '(no text)';

/**
 * True when a message's content would serialize to something strict providers
 * reject for the whole request: `null`/`undefined`, an empty string, an empty
 * array, or an array whose parts are all text parts with no visible text.
 * Non-text parts (images, files) make a turn usable by definition, so a
 * vision turn is never "empty". This is the shape a promptless send takes
 * when its attachments are no longer re-sent during history replay.
 */
function userContentHasNoVisibleText(
  content: string | MessageContentComplex[] | undefined | null
): boolean {
  if (content == null || content === '') {
    return true;
  }
  if (!Array.isArray(content)) {
    return false;
  }
  if (content.length === 0) {
    return true;
  }
  return content.every(
    (part) =>
      part != null &&
      part.type === ContentTypes.TEXT &&
      !getTextContent(part).trim()
  );
}

/**
 * True when an assistant message replayed as a steer and nothing followed it,
 * so the emitted run ends on the steer's `HumanMessage`.
 */
function endsWithSteerMessage(
  formatted: Array<RoleBearingMessage<BaseMessage>>
): boolean {
  if (formatted.length === 0) {
    return false;
  }
  return formatted[formatted.length - 1].additional_kwargs.source === 'steer';
}

interface MutableProviderMessageProvenancePart {
  attribution: ProviderMessageAttribution;
  sourceMessageId?: string;
  sourceContentPartIndices?: number[];
}

interface ProviderMessageProvenanceBuilder {
  readonly parts: MutableProviderMessageProvenancePart[];
  lastSeenSourceContentPartIndices?: Set<number>;
}

function createProviderMessageProvenanceBuilder(): ProviderMessageProvenanceBuilder {
  return { parts: [] };
}

function appendProviderProvenanceContribution(
  builder: ProviderMessageProvenanceBuilder,
  attribution: ProviderMessageAttribution,
  sourceMessageId?: string,
  sourceContentPartIndex?: number
): void {
  const last = builder.parts[builder.parts.length - 1];
  if (
    builder.parts.length === 0 ||
    last.attribution !== attribution ||
    last.sourceMessageId !== sourceMessageId ||
    (last.sourceContentPartIndices == null) !== (sourceContentPartIndex == null)
  ) {
    builder.parts.push({
      attribution,
      ...(sourceMessageId != null && { sourceMessageId }),
      ...(sourceContentPartIndex != null && {
        sourceContentPartIndices: [sourceContentPartIndex],
      }),
    });
    builder.lastSeenSourceContentPartIndices = undefined;
    return;
  }
  if (sourceContentPartIndex == null) {
    return;
  }
  const sourceContentPartIndices = (last.sourceContentPartIndices ??= []);
  if (sourceContentPartIndices.length === 0) {
    sourceContentPartIndices.push(sourceContentPartIndex);
    return;
  }
  const seen = (builder.lastSeenSourceContentPartIndices ??= new Set(
    sourceContentPartIndices
  ));
  if (!seen.has(sourceContentPartIndex)) {
    seen.add(sourceContentPartIndex);
    sourceContentPartIndices.push(sourceContentPartIndex);
  }
}

function appendProviderProvenanceIndices(
  builder: ProviderMessageProvenanceBuilder,
  attribution: ProviderMessageAttribution,
  sourceMessageId: string | undefined,
  sourceContentPartIndices?: SourceContentPartIndices,
  toolSourceContentPartIndices?: ReadonlySet<number>
): void {
  if (typeof sourceContentPartIndices === 'number') {
    appendProviderProvenanceContribution(
      builder,
      toolSourceContentPartIndices?.has(sourceContentPartIndices) === true
        ? 'tool'
        : attribution,
      sourceMessageId,
      sourceContentPartIndices
    );
    return;
  }
  if (
    sourceContentPartIndices == null ||
    sourceContentPartIndices.length === 0
  ) {
    appendProviderProvenanceContribution(builder, attribution, sourceMessageId);
    return;
  }
  for (const sourceContentPartIndex of sourceContentPartIndices) {
    appendProviderProvenanceContribution(
      builder,
      toolSourceContentPartIndices?.has(sourceContentPartIndex) === true
        ? 'tool'
        : attribution,
      sourceMessageId,
      sourceContentPartIndex
    );
  }
}

function appendSourceContentPartIndices(
  target: number[],
  sourceContentPartIndices: SourceContentPartIndices
): void {
  if (typeof sourceContentPartIndices === 'number') {
    target.push(sourceContentPartIndices);
  } else {
    for (const sourceContentPartIndex of sourceContentPartIndices) {
      target.push(sourceContentPartIndex);
    }
  }
}

function appendProviderProvenanceParts(
  builder: ProviderMessageProvenanceBuilder,
  parts: readonly ProviderMessageProvenancePart[]
): void {
  for (const part of parts) {
    appendProviderProvenanceIndices(
      builder,
      part.attribution,
      part.sourceMessageId,
      part.sourceContentPartIndices
    );
  }
}

function mergeAdjacentProviderProvenanceParts(
  parts: readonly ProviderMessageProvenancePart[]
): ProviderMessageProvenancePart[] {
  const builder = createProviderMessageProvenanceBuilder();
  appendProviderProvenanceParts(builder, parts);
  return builder.parts;
}

function createProviderContentProvenanceParts(
  content: readonly MessageContentComplex[],
  sourceContentPartOffset: number,
  sourceMessageId: string | undefined,
  defaultAttribution: ProviderMessageAttribution
): ProviderMessageProvenancePart[] {
  const builder = createProviderMessageProvenanceBuilder();
  for (let index = 0; index < content.length; index++) {
    const part = content[index] as MessageContentComplex | null | undefined;
    if (part == null) {
      continue;
    }
    appendProviderProvenanceContribution(
      builder,
      defaultAttribution,
      sourceMessageId,
      sourceContentPartOffset + index
    );
  }
  return builder.parts;
}

/**
 * Helper function to format an assistant message
 * @param message The message to format
 * @param options Optional formatting options
 * @returns Array of formatted messages
 */
function formatAssistantMessage(
  message: Partial<TMessage>,
  options?: FormatAssistantMessageOptions
): Array<
  | RoleBearingMessage<AIMessage>
  | RoleBearingMessage<ToolMessage>
  | RoleBearingMessage<HumanMessage>
> {
  const formattedMessages: Array<
    | RoleBearingMessage<AIMessage>
    | RoleBearingMessage<ToolMessage>
    | RoleBearingMessage<HumanMessage>
  > = [];
  const formattedMessageProvenance = new Map<
    BaseMessage,
    ProviderMessageProvenanceBuilder
  >();
  const appendFormattedMessage = (
    formattedMessage: (typeof formattedMessages)[number],
    attribution: ProviderMessageAttribution,
    sourceContentPartIndices?: SourceContentPartIndices,
    provenanceBuilder?: ProviderMessageProvenanceBuilder
  ): void => {
    const builder =
      provenanceBuilder ?? createProviderMessageProvenanceBuilder();
    if (provenanceBuilder == null) {
      appendProviderProvenanceIndices(
        builder,
        attribution,
        options?.sourceMessageId,
        sourceContentPartIndices
      );
    }
    formattedMessageProvenance.set(formattedMessage, builder);
    formattedMessages.push(formattedMessage);
  };
  const finalizeFormattedMessages = (): typeof formattedMessages => {
    for (let index = 0; index < formattedMessages.length; index++) {
      const formattedMessage = formattedMessages[index];
      stampSourceMessageIdentity(
        formattedMessage,
        options?.sourceMessageId,
        index,
        'model',
        undefined,
        formattedMessageProvenance.get(formattedMessage)?.parts
      );
    }
    return formattedMessages;
  };
  const appendSourcePartIndices = (
    formattedMessage: (typeof formattedMessages)[number],
    attribution: ProviderMessageAttribution,
    sourceContentPartIndices: SourceContentPartIndices
  ): void => {
    const builder = formattedMessageProvenance.get(formattedMessage);
    if (builder == null) {
      return;
    }
    appendProviderProvenanceIndices(
      builder,
      attribution,
      options?.sourceMessageId,
      sourceContentPartIndices
    );
  };
  const getSourcePartIndices = (partIndex: number): SourceContentPartIndices =>
    options?.sourceContentPartIndices?.[partIndex] ??
    (options?.sourceContentPartOffset ?? 0) + partIndex;
  const createSourceProvenanceBuilder = (
    sourceContentPartIndices: SourceContentPartIndices,
    attribution: ProviderMessageAttribution,
    applyToolOverrides = false
  ): ProviderMessageProvenanceBuilder => {
    const builder = createProviderMessageProvenanceBuilder();
    appendProviderProvenanceIndices(
      builder,
      attribution,
      options?.sourceMessageId,
      sourceContentPartIndices,
      applyToolOverrides ? options?.toolSourceContentPartIndices : undefined
    );
    return builder;
  };
  let currentContent: MessageContentComplex[] = [];
  let currentContentProvenance = createProviderMessageProvenanceBuilder();
  const appendCurrentContentProvenance = (
    sourceContentPartIndices: SourceContentPartIndices,
    attribution: ProviderMessageAttribution,
    applyToolOverrides = false
  ): void => {
    appendProviderProvenanceIndices(
      currentContentProvenance,
      attribution,
      options?.sourceMessageId,
      sourceContentPartIndices,
      applyToolOverrides ? options?.toolSourceContentPartIndices : undefined
    );
  };
  let lastAIMessage: RoleBearingMessage<AIMessage> | null = null;
  let hasReasoning = false;
  let pendingReasoningContent = '';
  let pendingReasoningSourcePartIndices: number[] = [];
  const emittedServerToolUseIds = new Set<string>();
  const pendingServerToolUses = new Map<
    string,
    {
      content: MessageContentComplex;
      sourceContentPartIndices: SourceContentPartIndices;
    }
  >();
  const shouldPreserveReasoningContent =
    options?.preserveReasoningContent === true;
  const compactionSemanticIndex = options?.compactionSemanticIndex;
  const semanticSourceMessageId =
    compactionSemanticIndex != null &&
    options?.sourceMessageId != null &&
    options.sourceMessageId.length <=
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars
      ? options.sourceMessageId
      : undefined;
  const retainedSourceContentStart = options?.sourceContentPartOffset ?? 0;
  const retainedSourceContentEnd =
    options?.retainedSourceContentEnd ?? retainedSourceContentStart;
  const serverToolResultIds = new Set<string>();
  const preferredToolCallParts = new Map<string, MessageContentComplex>();

  const takePendingReasoningContent = (): {
    content?: string;
    sourceContentPartIndices: number[];
  } => {
    if (!shouldPreserveReasoningContent || !pendingReasoningContent) {
      return { sourceContentPartIndices: [] };
    }
    const content = pendingReasoningContent;
    const sourceContentPartIndices = pendingReasoningSourcePartIndices;
    pendingReasoningContent = '';
    pendingReasoningSourcePartIndices = [];
    return { content, sourceContentPartIndices };
  };

  const createAIMessage = (
    content: MessageContent
  ): {
    message: RoleBearingMessage<AIMessage>;
    reasoningSourceContentPartIndices: number[];
  } => {
    const reasoning = takePendingReasoningContent();
    const message = withMessageRole(
      new AIMessage({
        content,
        ...(reasoning.content != null && {
          additional_kwargs: { reasoning_content: reasoning.content },
        }),
      }),
      'assistant'
    );
    return {
      message,
      reasoningSourceContentPartIndices: reasoning.sourceContentPartIndices,
    };
  };

  const appendAIMessage = (
    content: MessageContent,
    provenance: ProviderMessageProvenanceBuilder
  ): RoleBearingMessage<AIMessage> => {
    const created = createAIMessage(content);
    let combinedProvenance = provenance;
    if (created.reasoningSourceContentPartIndices.length > 0) {
      combinedProvenance = createProviderMessageProvenanceBuilder();
      appendProviderProvenanceIndices(
        combinedProvenance,
        'model',
        options?.sourceMessageId,
        created.reasoningSourceContentPartIndices
      );
      appendProviderProvenanceParts(combinedProvenance, provenance.parts);
    }
    appendFormattedMessage(
      created.message,
      'model',
      undefined,
      combinedProvenance
    );
    return created.message;
  };

  const attachPendingReasoningContent = (aiMessage: AIMessage): void => {
    const reasoning = takePendingReasoningContent();
    if (reasoning.content == null) {
      return;
    }
    aiMessage.additional_kwargs.reasoning_content =
      typeof aiMessage.additional_kwargs.reasoning_content === 'string'
        ? `${aiMessage.additional_kwargs.reasoning_content}${reasoning.content}`
        : reasoning.content;
    appendSourcePartIndices(
      aiMessage as (typeof formattedMessages)[number],
      'model',
      reasoning.sourceContentPartIndices
    );
  };

  const flushPendingServerToolUse = (toolUseId: string): void => {
    for (const [id, pending] of pendingServerToolUses) {
      pendingServerToolUses.delete(id);
      if (id === toolUseId) {
        currentContent.push(pending.content);
        appendCurrentContentProvenance(
          pending.sourceContentPartIndices,
          'model',
          true
        );
        emittedServerToolUseIds.add(id);
        return;
      }
    }
  };

  if (Array.isArray(message.content)) {
    const contentParts = message.content as Array<
      MessageContentComplex | undefined | null
    >;
    let trustedServerToolResultPartIndices: Set<number> | undefined;
    let wireServerToolResultPartIndices: Set<number> | undefined;

    for (let partIndex = 0; partIndex < contentParts.length; partIndex++) {
      const part = contentParts[partIndex];
      if (part == null) {
        continue;
      }
      if (part.type === ContentTypes.ACTIVITY_LABEL) {
        continue;
      }
      const isTrustedResult = isTrustedServerToolResult(
        part,
        getSourcePartIndices(partIndex),
        options?.toolSourceContentPartIndices
      );
      if (isTrustedResult) {
        trustedServerToolResultPartIndices ??= new Set();
        trustedServerToolResultPartIndices.add(partIndex);
      }
      if (isTrustedResult || isServerToolResultForWire(part)) {
        wireServerToolResultPartIndices ??= new Set();
        wireServerToolResultPartIndices.add(partIndex);
        serverToolResultIds.add(getToolUseId(part) ?? '');
      }
      if (options?.provider === Providers.ANTHROPIC) {
        const toolCallId = getToolCallId(part);
        if (toolCallId == null) {
          continue;
        }
        const preferredPart = preferredToolCallParts.get(toolCallId);
        if (
          preferredPart == null ||
          (!hasToolCallOutput(preferredPart) && hasToolCallOutput(part))
        ) {
          preferredToolCallParts.set(toolCallId, part);
        }
      }
    }

    for (let partIndex = 0; partIndex < contentParts.length; partIndex++) {
      const part = contentParts[partIndex];
      if (part == null) {
        continue;
      }
      const sourcePartIndices = getSourcePartIndices(partIndex);
      if (part.type === ContentTypes.ACTIVITY_LABEL) {
        if (
          compactionSemanticIndex != null &&
          semanticSourceMessageId != null &&
          typeof sourcePartIndices === 'number'
        ) {
          collectCompactionSemanticEntriesFromPart(
            compactionSemanticIndex,
            part,
            semanticSourceMessageId,
            sourcePartIndices,
            retainedSourceContentStart,
            retainedSourceContentEnd,
            options?.intentToolNames
          );
        }
        continue;
      }
      const toolUseId = getToolUseId(part);
      if (toolUseId != null) {
        const isServerToolResult =
          trustedServerToolResultPartIndices?.has(partIndex) === true;
        const isWireServerToolResult =
          wireServerToolResultPartIndices?.has(partIndex) === true;
        flushPendingServerToolUse(toolUseId);
        if (isWireServerToolResult) {
          currentContent.push(part);
          appendCurrentContentProvenance(
            sourcePartIndices,
            isServerToolResult ? 'tool' : 'model',
            !isServerToolResult
          );
          continue;
        }
      } else if (hasMeaningfulAssistantContent(part)) {
        for (const id of pendingServerToolUses.keys()) {
          if (!serverToolResultIds.has(id)) {
            pendingServerToolUses.delete(id);
          }
        }
      }
      if (part.type === ContentTypes.TEXT && part.tool_call_ids) {
        /*
        If there's pending content, it needs to be aggregated as a single string to prepare for tool calls.
        For Anthropic models, the "tool_calls" field on a message is only respected if content is a string.
        */
        if (currentContent.length > 0) {
          if (
            currentContent.some((content) => content.type !== ContentTypes.TEXT)
          ) {
            currentContent.push(part);
            appendCurrentContentProvenance(sourcePartIndices, 'model', true);
            lastAIMessage = appendAIMessage(
              toLangChainContent(currentContent),
              currentContentProvenance
            );
            currentContent = [];
            currentContentProvenance = createProviderMessageProvenanceBuilder();
            continue;
          }
          let content = currentContent.reduce((acc, curr) => {
            if (curr.type === ContentTypes.TEXT) {
              return `${acc}${getTextContent(curr)}\n`;
            }
            return acc;
          }, '');
          content = `${content}\n${getTextContent(part)}`.trim();
          appendCurrentContentProvenance(sourcePartIndices, 'model', true);
          lastAIMessage = appendAIMessage(content, currentContentProvenance);
          currentContent = [];
          currentContentProvenance = createProviderMessageProvenanceBuilder();
          continue;
        }
        // Create a new AIMessage with this text and prepare for tool calls
        lastAIMessage = appendAIMessage(
          getTextContent(part),
          createSourceProvenanceBuilder(sourcePartIndices, 'model', true)
        );
      } else if (part.type === ContentTypes.TOOL_CALL) {
        // Skip malformed tool call entries without tool_call property
        if (part.tool_call == null) {
          continue;
        }
        const toolCallId = getToolCallId(part);
        if (
          options?.provider === Providers.ANTHROPIC &&
          toolCallId != null &&
          preferredToolCallParts.get(toolCallId) !== part
        ) {
          continue;
        }

        // Note: `tool_calls` list is defined when constructed by `AIMessage` class, and outputs should be excluded from it
        const {
          output,
          args: _args,
          ..._tool_call
        } = part.tool_call as ToolCallPart;

        // Skip invalid tool calls that have no name AND no output
        if (
          _tool_call.name == null ||
          (_tool_call.name === '' && (output == null || output === ''))
        ) {
          continue;
        }

        if (
          options?.provider === Providers.ANTHROPIC &&
          typeof _tool_call.id === 'string' &&
          _tool_call.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
        ) {
          if (
            !serverToolResultIds.has(_tool_call.id) &&
            options.preserveUnpairedServerToolUses !== true
          ) {
            continue;
          }
          if (
            emittedServerToolUseIds.has(_tool_call.id) ||
            pendingServerToolUses.has(_tool_call.id)
          ) {
            continue;
          }
          const serverToolInput = parseServerToolInput(_args);
          if (
            compactionSemanticIndex != null &&
            semanticSourceMessageId != null &&
            typeof sourcePartIndices === 'number'
          ) {
            collectCompactionSemanticEntriesFromPart(
              compactionSemanticIndex,
              part,
              semanticSourceMessageId,
              sourcePartIndices,
              retainedSourceContentStart,
              retainedSourceContentEnd,
              options.intentToolNames,
              serverToolInput
            );
          }
          pendingServerToolUses.set(_tool_call.id, {
            content: {
              type: 'server_tool_use',
              id: _tool_call.id,
              name: _tool_call.name,
              input: serverToolInput,
            } as MessageContentComplex,
            sourceContentPartIndices: sourcePartIndices,
          });
          continue;
        }

        if (!lastAIMessage) {
          // "Heal" the payload by creating an AIMessage to precede the tool call
          lastAIMessage = appendAIMessage(
            '',
            createSourceProvenanceBuilder(sourcePartIndices, 'model')
          );
        } else {
          attachPendingReasoningContent(lastAIMessage);
          appendSourcePartIndices(lastAIMessage, 'model', sourcePartIndices);
        }

        const tool_call: ToolCallPart = _tool_call;
        // TODO: investigate; args as dictionary may need to be providers-or-tool-specific
        let args: any = _args;
        try {
          if (typeof _args === 'string') {
            args = JSON.parse(_args);
          }
        } catch {
          if (typeof _args === 'string') {
            args = { input: _args };
          }
        }

        tool_call.args = args;
        if (
          compactionSemanticIndex != null &&
          semanticSourceMessageId != null &&
          typeof sourcePartIndices === 'number'
        ) {
          collectCompactionSemanticEntriesFromPart(
            compactionSemanticIndex,
            part,
            semanticSourceMessageId,
            sourcePartIndices,
            retainedSourceContentStart,
            retainedSourceContentEnd,
            options?.intentToolNames,
            args
          );
        }
        if (
          options?.provider === Providers.ANTHROPIC &&
          Array.isArray(lastAIMessage.content)
        ) {
          const content = lastAIMessage.content as MessageContentComplex[];
          content.push({
            type: 'tool_use',
            id: normalizeAnthropicToolCallId(tool_call.id ?? ''),
            name: tool_call.name,
            input: args,
          } as MessageContentComplex);
          lastAIMessage.content = content as MessageContent;
        } else {
          if (!lastAIMessage.tool_calls) {
            lastAIMessage.tool_calls = [];
          }
          lastAIMessage.tool_calls.push(tool_call as ToolCall);
        }

        appendFormattedMessage(
          withMessageRole(
            new ToolMessage({
              tool_call_id: tool_call.id ?? '',
              name: tool_call.name,
              content: formatToolCallOutput(output),
            }),
            'tool'
          ),
          'tool',
          sourcePartIndices
        );
      } else if (isReasoningContentBlock(part)) {
        if (
          part.type === ContentTypes.THINK &&
          compactionSemanticIndex != null &&
          semanticSourceMessageId != null &&
          typeof sourcePartIndices === 'number'
        ) {
          collectCompactionSemanticEntriesFromPart(
            compactionSemanticIndex,
            part,
            semanticSourceMessageId,
            sourcePartIndices,
            retainedSourceContentStart,
            retainedSourceContentEnd,
            options?.intentToolNames
          );
        }
        hasReasoning = true;
        pendingReasoningContent += extractReasoningContent(part);
        appendSourceContentPartIndices(
          pendingReasoningSourcePartIndices,
          sourcePartIndices
        );
        continue;
      } else if (part.type === ContentTypes.STEER) {
        /*
        A mid-run steer: user speech persisted inline in the assistant message
        at the tool-batch boundary it was injected. Flush accumulated
        assistant content first so ordering is preserved, then replay the
        steer as a standalone user message — multimodal when the host stamped
        a pre-encoded `media` content array (attachment refs are re-encoded
        per turn host-side, like any other user media). `lastAIMessage` is
        reset AFTER the HumanMessage: a post-steer tool_call must mint a
        FRESH assistant anchor (the heal path) so its AIMessage lands after
        the user turn — attaching it to the pre-steer anchor would emit its
        ToolMessage after the HumanMessage while the call itself sat before
        it, an invalid provider ordering.
        */
        if (currentContent.length > 0) {
          if (
            currentContent.some((content) => content.type !== ContentTypes.TEXT)
          ) {
            appendAIMessage(
              toLangChainContent(currentContent),
              currentContentProvenance
            );
          } else {
            const flushed = currentContent
              .reduce((acc, curr) => `${acc}${getTextContent(curr)}\n`, '')
              .trim();
            if (flushed.length > 0) {
              appendAIMessage(flushed, currentContentProvenance);
            }
          }
          currentContent = [];
          currentContentProvenance = createProviderMessageProvenanceBuilder();
        } else if (shouldPreserveReasoningContent && pendingReasoningContent) {
          /**
           * Reasoning directly preceding a steer has no `currentContent`
           * flush to consume it and the anchor resets below — emit an anchor
           * AIMessage now (createAIMessage folds the pending reasoning into
           * `additional_kwargs.reasoning_content`) or the persisted
           * assistant reasoning silently vanishes on replay.
           */
          appendAIMessage('', createProviderMessageProvenanceBuilder());
        }
        const steerPart = part as {
          steer?: string;
          media?: MessageContentComplex[];
        };
        const steerContent =
          Array.isArray(steerPart.media) && steerPart.media.length > 0
            ? toLangChainContent(steerPart.media)
            : (steerPart.steer ?? '');
        appendFormattedMessage(
          withMessageRole(
            new HumanMessage({
              content: steerContent as MessageContent,
              additional_kwargs: { role: 'user', source: 'steer' },
            }),
            'user'
          ),
          'user',
          sourcePartIndices
        );
        lastAIMessage = null;
        /** The steer splits the assistant message: the post-steer segment
         *  starts with fresh reasoning state (pre-steer reasoning was either
         *  flushed above or intentionally dropped when not preserving). */
        hasReasoning = false;
        pendingReasoningContent = '';
        pendingReasoningSourcePartIndices = [];
      } else if (
        part.type === ContentTypes.ERROR ||
        part.type === ContentTypes.AGENT_UPDATE ||
        part.type === ContentTypes.SUMMARY
      ) {
        continue;
      } else {
        if (part.type === ContentTypes.TEXT && !getTextContent(part).trim()) {
          continue;
        }
        currentContent.push(part);
        appendCurrentContentProvenance(sourcePartIndices, 'model', true);
      }
    }
    for (const pending of pendingServerToolUses.values()) {
      currentContent.push(pending.content);
      appendCurrentContentProvenance(
        pending.sourceContentPartIndices,
        'model',
        true
      );
    }
  }

  if (hasReasoning && currentContent.length > 0) {
    let content = '';
    for (const part of currentContent) {
      if (part.type !== ContentTypes.TEXT) {
        appendAIMessage(
          toLangChainContent(currentContent),
          currentContentProvenance
        );
        return finalizeFormattedMessages();
      }
      content += `${getTextContent(part)}\n`;
    }
    content = content.trim();

    if (content) {
      appendAIMessage(content, currentContentProvenance);
    }
  } else if (currentContent.length > 0) {
    appendAIMessage(
      toLangChainContent(currentContent),
      currentContentProvenance
    );
  }

  return finalizeFormattedMessages();
}

function getSourceMessageId(message: Partial<TMessage>): string | undefined {
  const candidates = [
    (message as { messageId?: unknown }).messageId,
    (message as { id?: unknown }).id,
  ];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') {
      continue;
    }
    const normalized = candidate.trim();
    if (normalized.length > 0) {
      return normalized;
    }
  }
  return undefined;
}

/**
 * Keeps the first formatted message backward-compatible with its persisted
 * source id and preserves source correlation on every derived message.
 * Derived ids remain unset so the reducer can assign collision-free identities
 * during its existing pass. This also prevents invented provider-shaped ids
 * from leaking into provider request payloads.
 */
function stampSourceMessageIdentity(
  message: RoleBearingMessage<BaseMessage>,
  sourceMessageId: string | undefined,
  derivedIndex = 0,
  attribution: ProviderMessageAttribution = 'model',
  sourceContentPartIndices?: readonly number[],
  provenanceParts?: readonly ProviderMessageProvenancePart[]
): void {
  if (sourceMessageId != null) {
    message.additional_kwargs.sourceMessageId = sourceMessageId;
  }
  let partsToStamp: readonly ProviderMessageProvenancePart[];
  if (provenanceParts != null && provenanceParts.length > 0) {
    let needsSourceMessageId = false;
    if (sourceMessageId != null) {
      for (const part of provenanceParts) {
        if (part.sourceMessageId == null) {
          needsSourceMessageId = true;
          break;
        }
      }
    }
    if (needsSourceMessageId) {
      const completedParts: ProviderMessageProvenancePart[] = [];
      for (const part of provenanceParts) {
        completedParts.push({
          ...part,
          ...(part.sourceMessageId == null && { sourceMessageId }),
        });
      }
      partsToStamp = completedParts;
    } else {
      partsToStamp = provenanceParts;
    }
  } else {
    partsToStamp = [
      {
        attribution,
        ...(sourceMessageId != null && { sourceMessageId }),
        ...(sourceContentPartIndices != null && {
          sourceContentPartIndices,
        }),
      },
    ];
  }
  setFreshProviderMessageProvenance(message, partsToStamp);
  if (sourceMessageId == null || derivedIndex !== 0) {
    return;
  }
  message.id = sourceMessageId;
  message.lc_kwargs.id = sourceMessageId;
}

/**
 * Labels all agent content for parallel patterns (fan-out/fan-in)
 * Groups consecutive content by agent and wraps with clear labels
 */
function labelAllAgentContent(
  contentParts: MessageContentComplex[],
  agentIdMap: Record<number, string>,
  agentNames?: Record<string, string>
): MessageContentComplex[] {
  const result: MessageContentComplex[] = [];
  let currentAgentId: string | undefined;
  let agentContentBuffer: MessageContentComplex[] = [];

  const flushAgentBuffer = (): void => {
    if (agentContentBuffer.length === 0) {
      return;
    }

    if (currentAgentId != null && currentAgentId !== '') {
      const agentName = (agentNames?.[currentAgentId] ?? '') || currentAgentId;
      const formattedParts: string[] = [];

      formattedParts.push(`--- ${agentName} ---`);

      for (const part of agentContentBuffer) {
        if (part.type === ContentTypes.THINK) {
          const thinkContent = (part as ReasoningContentText).think || '';
          if (thinkContent) {
            formattedParts.push(
              `${agentName}: ${JSON.stringify({
                type: 'think',
                think: thinkContent,
              })}`
            );
          }
        } else if (part.type === ContentTypes.TEXT) {
          const textContent: string = part.text ?? '';
          if (textContent) {
            formattedParts.push(`${agentName}: ${textContent}`);
          }
        } else if (part.type === ContentTypes.TOOL_CALL) {
          formattedParts.push(
            `${agentName}: ${JSON.stringify({
              type: 'tool_call',
              tool_call: (part as ToolCallContent).tool_call,
            })}`
          );
        }
      }

      formattedParts.push(`--- End of ${agentName} ---`);

      // Create a single text content part with all agent content
      result.push({
        type: ContentTypes.TEXT,
        text: formattedParts.join('\n\n'),
      } as MessageContentComplex);
    } else {
      // No agent ID, pass through as-is
      for (const part of agentContentBuffer) {
        result.push(part);
      }
    }

    agentContentBuffer = [];
  };

  for (let i = 0; i < contentParts.length; i++) {
    const part = contentParts[i];
    /** UI-only progress headers are not agent content and must not disturb
     *  agent state: a label with no `agentIdMap` entry would otherwise read
     *  as an agent change and flush the buffer mid-agent, splitting one
     *  agent's contiguous content into two labeled blocks. Skipped before
     *  any state transition below (mirrors the transfer path). */
    if (part.type === ContentTypes.ACTIVITY_LABEL) {
      continue;
    }
    const agentId = agentIdMap[i];

    // If agent changed, flush previous buffer
    if (agentId !== currentAgentId && currentAgentId !== undefined) {
      flushAgentBuffer();
    }

    currentAgentId = agentId;
    if (part.type === ContentTypes.STEER) {
      /** User speech is never agent content — see the transfer path above. */
      flushAgentBuffer();
      result.push(part);
      continue;
    }
    agentContentBuffer.push(part);
  }

  // Flush any remaining content
  flushAgentBuffer();

  return result;
}

/**
 * Groups content parts by agent and formats them with agent labels
 * This preprocesses multi-agent content to prevent identity confusion
 *
 * @param contentParts - The content parts from a run
 * @param agentIdMap - Map of content part index to agent ID
 * @param agentNames - Optional map of agent ID to display name
 * @param options - Configuration options
 * @param options.labelNonTransferContent - If true, labels all agent transitions (for parallel patterns)
 * @returns Modified content parts with agent labels where appropriate
 */
export const labelContentByAgent = (
  contentParts: MessageContentComplex[],
  agentIdMap?: Record<number, string>,
  agentNames?: Record<string, string>,
  options?: { labelNonTransferContent?: boolean }
): MessageContentComplex[] => {
  if (!agentIdMap || Object.keys(agentIdMap).length === 0) {
    return contentParts;
  }

  // If labelNonTransferContent is true, use a different strategy for parallel patterns
  if (options?.labelNonTransferContent === true) {
    return labelAllAgentContent(contentParts, agentIdMap, agentNames);
  }

  const result: MessageContentComplex[] = [];
  let currentAgentId: string | undefined;
  let agentContentBuffer: MessageContentComplex[] = [];
  let transferToolCallIndex: number | undefined;
  let transferToolCallId: string | undefined;

  const flushAgentBuffer = (): void => {
    if (agentContentBuffer.length === 0) {
      return;
    }

    // If this is content from a transferred agent, format it specially
    if (
      currentAgentId != null &&
      currentAgentId !== '' &&
      transferToolCallIndex !== undefined
    ) {
      const agentName = (agentNames?.[currentAgentId] ?? '') || currentAgentId;
      const formattedParts: string[] = [];

      formattedParts.push(`--- Transfer to ${agentName} ---`);

      for (const part of agentContentBuffer) {
        if (part.type === ContentTypes.THINK) {
          formattedParts.push(
            `${agentName}: ${JSON.stringify({
              type: 'think',
              think: (part as ReasoningContentText).think,
            })}`
          );
        } else if ('text' in part && part.type === ContentTypes.TEXT) {
          const textContent: string = part.text ?? '';
          if (textContent) {
            formattedParts.push(
              `${agentName}: ${JSON.stringify({
                type: 'text',
                text: textContent,
              })}`
            );
          }
        } else if (part.type === ContentTypes.TOOL_CALL) {
          formattedParts.push(
            `${agentName}: ${JSON.stringify({
              type: 'tool_call',
              tool_call: (part as ToolCallContent).tool_call,
            })}`
          );
        }
      }

      formattedParts.push(`--- End of ${agentName} response ---`);

      // Find the tool call that triggered this transfer and update its output
      if (transferToolCallIndex < result.length) {
        const transferToolCall = result[transferToolCallIndex];
        if (
          transferToolCall.type === ContentTypes.TOOL_CALL &&
          transferToolCall.tool_call?.id === transferToolCallId
        ) {
          transferToolCall.tool_call.output = formattedParts.join('\n\n');
        }
      }
    } else {
      // Not from a transfer, add as-is
      for (const part of agentContentBuffer) {
        result.push(part);
      }
    }

    agentContentBuffer = [];
    transferToolCallIndex = undefined;
    transferToolCallId = undefined;
  };

  for (let i = 0; i < contentParts.length; i++) {
    const part = contentParts[i];
    /** UI-only progress headers are not agent content and must not disturb
     *  agent state: a label with no `agentIdMap` entry would otherwise look
     *  like an agent change, flushing the buffer and resetting an open
     *  transfer capture so the transferred agent's following chunks lose
     *  their frame. Skipped before any state transition below. */
    if (part.type === ContentTypes.ACTIVITY_LABEL) {
      continue;
    }
    const agentId = agentIdMap[i];

    // Check if this is a transfer tool call
    const isTransferTool =
      (part.type === ContentTypes.TOOL_CALL &&
        (part as ToolCallContent).tool_call?.name?.startsWith(
          'lc_transfer_to_'
        )) ??
      false;

    // If agent changed, flush previous buffer
    if (agentId !== currentAgentId && currentAgentId !== undefined) {
      flushAgentBuffer();
    }

    currentAgentId = agentId;

    if (isTransferTool) {
      // Flush any existing buffer first
      flushAgentBuffer();
      // Add the transfer tool call to result
      result.push(part);
      // Mark that the next agent's content should be captured
      transferToolCallIndex = result.length - 1;
      transferToolCallId = (part as ToolCallContent).tool_call?.id;
      currentAgentId = undefined; // Reset to capture the next agent
    } else if (part.type === ContentTypes.STEER) {
      /**
       * User speech is never agent content: flush the buffer in place and
       * pass the steer through verbatim so `formatAssistantMessage` replays
       * it as a user turn. Folding it into a labeled transfer summary would
       * both drop the user's words and misattribute them to the agent.
       * The steer also CLOSES any open transfer capture — `flushAgentBuffer`
       * no-ops on an empty buffer, and leaving the capture live would fold
       * post-steer agent content into the pre-steer transfer output,
       * replaying it BEFORE the user's redirect.
       */
      flushAgentBuffer();
      transferToolCallIndex = undefined;
      transferToolCallId = undefined;
      currentAgentId = undefined;
      result.push(part);
    } else {
      agentContentBuffer.push(part);
    }
  }

  flushAgentBuffer();

  return result;
};

/** Extracts tool names from a tool_search output JSON string. */
function extractToolNamesFromSearchOutput(output: string): string[] {
  try {
    const parsed: unknown = JSON.parse(output);
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      Array.isArray((parsed as Record<string, unknown>).tools)
    ) {
      return (
        (parsed as Record<string, unknown>).tools as Array<{ name?: string }>
      )
        .map((t) => t.name)
        .filter((name): name is string => typeof name === 'string');
    }
  } catch {
    /** Output may have warnings prepended, try to find JSON within it */
    const jsonMatch = output.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        const parsed: unknown = JSON.parse(jsonMatch[0]);
        if (
          typeof parsed === 'object' &&
          parsed !== null &&
          Array.isArray((parsed as Record<string, unknown>).tools)
        ) {
          return (
            (parsed as Record<string, unknown>).tools as Array<{
              name?: string;
            }>
          )
            .map((t) => t.name)
            .filter((name): name is string => typeof name === 'string');
        }
      } catch {
        /* ignore */
      }
    }
  }
  return [];
}

/**
 * How far back a persisted summary reaches.
 *
 * `coverage` is authoritative: the block named the first source message that
 * compaction retained, so `messageIndex` is exclusive — everything before it is
 * covered and it survives whole. `positional` is the legacy reading for blocks
 * written before coverage existed (or whose anchor is no longer in the payload)
 * — the block's own location is the boundary, which is why it cannot
 * distinguish a retained tail from covered history.
 */
type SummaryBoundary =
  | {
      mode: 'coverage';
      messageIndex: number;
      text: string;
      tokenCount: number;
    }
  | {
      mode: 'positional';
      messageIndex: number;
      contentIndex: number;
      text: string;
      tokenCount: number;
    };

type SummaryTokenAdjustment = {
  original: number;
  adjusted: number;
  remainingChars: number;
  totalChars: number;
};

type SummaryScan = {
  boundary?: SummaryBoundary;
};

function resolveCoverageIndex(
  coverage: SummaryCoverage | undefined,
  indexBySourceId: Map<string, number>,
  summaryMessageIndex: number
): number | undefined {
  /** Persisted JSON, so the declared string type is not a runtime guarantee. */
  if (typeof coverage?.retainedFromMessageId !== 'string') {
    return undefined;
  }
  const retainedFromMessageId = coverage.retainedFromMessageId.trim();
  if (retainedFromMessageId === '') {
    return undefined;
  }
  const retainedIndex = indexBySourceId.get(retainedFromMessageId);
  return retainedIndex != null && retainedIndex <= summaryMessageIndex
    ? retainedIndex
    : undefined;
}

function scanSummaryBlocks(payload: TPayload): SummaryScan {
  let boundary: SummaryBoundary | undefined;
  /** Filled as the scan advances, so a coverage lookup only ever resolves to a
   *  message already passed — no second pass over the payload. */
  const indexBySourceId = new Map<string, number>();

  for (let i = 0; i < payload.length; i++) {
    const message = payload[i];
    const sourceMessageId = getSourceMessageId(message);
    if (sourceMessageId != null) {
      indexBySourceId.set(sourceMessageId, i);
    }
    if (!Array.isArray(message.content)) {
      continue;
    }

    for (let j = 0; j < message.content.length; j++) {
      const part = message.content[j] as MessageContentComplex | undefined;
      if (part == null || part.type !== ContentTypes.SUMMARY) {
        continue;
      }

      const summaryPart = part as Partial<SummaryContentBlock> & {
        text?: string;
      };

      // Try content array first (new format), then direct text (legacy format)
      let summaryText = (summaryPart.content ?? [])
        .map((block) =>
          'text' in block ? (block as { text: string }).text : ''
        )
        .join('')
        .trim();

      // Fallback: legacy format where text was a direct field on the block
      if (summaryText.length === 0 && typeof summaryPart.text === 'string') {
        summaryText = summaryPart.text.trim();
      }

      if (summaryText.length === 0) {
        continue;
      }

      const tokenCount =
        typeof summaryPart.tokenCount === 'number' &&
        Number.isFinite(summaryPart.tokenCount)
          ? summaryPart.tokenCount
          : 0;

      const retainedIndex = resolveCoverageIndex(
        summaryPart.coverage,
        indexBySourceId,
        i
      );

      boundary =
        retainedIndex != null
          ? {
            mode: 'coverage',
            messageIndex: retainedIndex,
            text: summaryText,
            tokenCount,
          }
          : {
            mode: 'positional',
            messageIndex: i,
            contentIndex: j,
            text: summaryText,
            tokenCount,
          };
    }
  }

  return { boundary };
}

function applySummaryBoundary(
  message: Partial<TMessage>,
  messageIndex: number,
  summaryBoundary?: SummaryBoundary
): Partial<TMessage> | null {
  if (!summaryBoundary) {
    return message;
  }

  /** The boundary names the first retained message, so it is exclusive: that
   *  message and everything after it — the recency tail included — stays
   *  verbatim, and only genuinely covered history is dropped. Summary parts on
   *  surviving messages are filtered later by `formatAssistantMessage`. */
  if (summaryBoundary.mode === 'coverage') {
    return messageIndex < summaryBoundary.messageIndex ? null : message;
  }

  if (messageIndex < summaryBoundary.messageIndex) {
    return null;
  }

  if (
    messageIndex !== summaryBoundary.messageIndex ||
    !Array.isArray(message.content)
  ) {
    return message;
  }

  return {
    ...message,
    content: message.content.slice(summaryBoundary.contentIndex + 1),
  };
}

/**
 * Whether `formatAssistantMessage` filters this part out of the emitted message.
 * Such a part contributes no prompt tokens, so measuring it as zero characters
 * is accurate — it must not be mistaken for content the heuristic cannot see.
 */
function isDroppedByFormatting(
  part: MessageContentComplex | undefined
): boolean {
  if (part == null) {
    return true;
  }
  if (
    part.type === ContentTypes.SUMMARY ||
    part.type === ContentTypes.ERROR ||
    part.type === ContentTypes.AGENT_UPDATE ||
    part.type === ContentTypes.ACTIVITY_LABEL
  ) {
    return true;
  }
  return part.type === ContentTypes.TEXT && getTextContent(part).trim() === '';
}

function measureValueChars(value: unknown): number {
  if (typeof value === 'string') {
    return value.length;
  }
  if (value == null || typeof value !== 'object') {
    return 0;
  }
  const measured = serializeStructuredValueBounded(value, 0).originalChars;
  return measured === Number.MAX_SAFE_INTEGER
    ? HARD_MAX_TOOL_RESULT_CHARS
    : Math.min(measured, HARD_MAX_TOOL_RESULT_CHARS);
}

/**
 * Whether a retained part's whole prompt cost is the text the char heuristic
 * reads, making it safe to represent in a character ratio.
 *
 * An allowlist, not a denylist. Media, resources, and tool calls carry cost that
 * is unrelated to their serialized length — a short image URL nested in
 * `tool_call.output` stands in for a fixed four-figure media charge — and
 * rejecting those case by case has repeatedly missed a nesting level. Listing
 * the two shapes whose characters `contentPartCharLength` actually reads makes
 * every other shape, present or future, ineligible by default: the ratio is
 * skipped and the entry keeps its original count, which prunes early rather than
 * exceeding the window.
 */
function isCharRatioEligible(part: MessageContentComplex | undefined): boolean {
  if (part == null) {
    return false;
  }
  return part.type === ContentTypes.TEXT || part.type === ContentTypes.THINKING;
}

function contentPartCharLength(part: MessageContentComplex): number {
  const record = part as Record<string, unknown>;
  let len = 0;
  if (typeof record.text === 'string') {
    len += record.text.length;
  }
  if (typeof record.thinking === 'string') {
    len += record.thinking.length;
  }
  len += measureValueChars(record.input);
  /** Tool calls nest their payload a level down, so measuring only the
   *  top-level fields scores an entire tool turn as zero characters. */
  const { tool_call: toolCall } = record;
  if (toolCall != null && typeof toolCall === 'object') {
    const call = toolCall as Record<string, unknown>;
    len += measureValueChars(call.name);
    len += measureValueChars(call.args);
    len += measureValueChars(call.output);
  }
  return len;
}

/** Extracts the skillName from a skill tool_call's args (string or object). */
function extractSkillName(args: unknown): string | undefined {
  let parsed: Record<string, unknown> | undefined;
  if (typeof args === 'string') {
    try {
      parsed = JSON.parse(args) as Record<string, unknown>;
    } catch {
      /* malformed args — skip */
    }
  } else {
    parsed = args as Record<string, unknown> | undefined;
  }
  const name = parsed?.skillName;
  return typeof name === 'string' && name !== '' ? name : undefined;
}

function isSdkManagedToolName(name: unknown): name is string {
  return (
    typeof name === 'string' &&
    (name === Constants.SUBAGENT ||
      name === 'conditional_transfer' ||
      name.startsWith(Constants.LC_TRANSFER_TO_))
  );
}

/**
 * Formats an array of messages for LangChain, handling tool calls and creating ToolMessage instances.
 *
 * @param payload - The array of messages to format.
 * @param indexTokenCountMap - Optional map of message indices to token counts.
 * @param tools - Optional set of host-managed tool names allowed in the request. SDK-managed tool
 *   names are preserved independently.
 * @param skills - Optional map of skill name to body for reconstructing skill HumanMessages.
 * @param options - Optional formatting options (provider, skipSkillBodyNames).
 * @returns - Object containing formatted messages and updated indexTokenCountMap if provided.
 */
export const formatAgentMessages = (
  payload: TPayload,
  indexTokenCountMap?: Record<number, number | undefined>,
  tools?: Set<string>,
  /** Pre-resolved skill bodies keyed by skill name. When present, HumanMessages
   *  are reconstructed after skill ToolMessages to restore skill instructions
   *  that were only in LangGraph state during the original run. */
  skills?: Map<string, string>,
  options?: FormatAgentMessagesOptions
): {
  messages: Array<
    | RoleBearingMessage<HumanMessage>
    | RoleBearingMessage<AIMessage>
    | RoleBearingMessage<SystemMessage>
    | RoleBearingMessage<ToolMessage>
  >;
  indexTokenCountMap?: Record<number, number>;
  /** Cross-run summary extracted from the payload. Should be forwarded to the
   *  agent run so it can be included in the system message via AgentContext. */
  summary?: { text: string; tokenCount: number };
  /** When a positional summary boundary sliced content from a message, the token
   *  count was proportionally reduced. Returned so the caller can log it. */
  boundaryTokenAdjustment?: SummaryTokenAdjustment;
  /** Bounded semantic guidance derived during persisted-content analysis. */
  compactionSemanticIndex?: CompactionSemanticIndex;
  /** Serializable continuation state for incrementally evolving the index. */
  compactionSemanticIndexSnapshot?: CompactionSemanticIndexSnapshot;
} => {
  const messages: Array<
    | RoleBearingMessage<HumanMessage>
    | RoleBearingMessage<AIMessage>
    | RoleBearingMessage<SystemMessage>
    | RoleBearingMessage<ToolMessage>
  > = [];
  /**
   * A steer ended the previous payload entry, so the next message emitted —
   * whichever entry finally produces one — must be separated from it by an
   * assistant turn. Held rather than emitted so an entry that produces
   * nothing cannot leave the anchor stranded as the final turn.
   */
  const legacyContentEnabled = options?.legacyContent === true;
  const compactionSemanticIndexCollector =
    options?.compactionSemanticIndex != null
      ? createDerivedCompactionSemanticIndexCollector(
        options.compactionSemanticIndex.baseSnapshot
      )
      : undefined;
  /** Emission choke point: every formatted message enters the result here, so
   *  the legacy flatten happens once per message with no closing rescan. The
   *  summary boundary slices payload entries before formatting, and nothing
   *  mutates an emitted message's content afterwards, so flattening at
   *  emission and flattening at return are equivalent. */
  const emitFormattedMessage = (message: (typeof messages)[number]): void => {
    if (legacyContentEnabled && isLegacyConvertible(message)) {
      const flattened = flattenLegacyContent(
        message.content as MessageContentComplex[]
      );
      message.content = flattened;
      message.lc_kwargs.content = flattened;
    }
    messages.push(message);
  };
  let pendingSteerAnchor = false;
  /**
   * Emits the deferred anchor ahead of `next` — the message about to be
   * pushed. When that message is itself an assistant turn, it already IS the
   * separation the anchor exists to synthesize, so the intent is simply
   * discharged: emitting the placeholder anyway would put two assistant turns
   * back to back, which strict-alternation providers can reject and nothing downstream
   * repairs (`coalesceAdjacentUserTurns` merges user turns only).
   */
  const flushSteerAnchor = (next: { role?: LangChainMessageRole }): void => {
    if (!pendingSteerAnchor) {
      return;
    }
    pendingSteerAnchor = false;
    if (next.role === 'assistant') {
      return;
    }
    const anchor = withMessageRole(
      new AIMessage({ content: STEER_ANCHOR_PLACEHOLDER }),
      'assistant'
    );
    stampSourceMessageIdentity(anchor, undefined, 0, 'synthetic');
    emitFormattedMessage(anchor);
  };
  // If indexTokenCountMap is provided, create a new map to track the updated indices
  const updatedIndexTokenCountMap: Record<number, number> = {};
  let boundaryTokenAdjustment: SummaryTokenAdjustment | undefined;
  // Keep track of the mapping from original payload indices to result indices
  const indexMapping: Record<number, number[] | undefined> = {};
  const { boundary: summaryBoundary } = scanSummaryBlocks(payload);

  // Summary metadata is returned to the caller so it can be forwarded to the
  // agent run and included in the single system message via AgentContext.
  // We intentionally do NOT create a SystemMessage here — that would conflict
  // with the agent's own system message (instructions + summary combined).

  /**
   * Create a mutable copy of the tools set that can be expanded dynamically.
   * When we encounter tool_search results, we add discovered tools to this set,
   * making their subsequent tool calls valid.
   */
  const discoveredTools = tools ? new Set(tools) : undefined;

  // Process messages with tool conversion if tools set is provided
  for (let i = 0; i < payload.length; i++) {
    const rawMessage = payload[i];
    const sourceMessageId = getSourceMessageId(rawMessage);
    let message = applySummaryBoundary(rawMessage, i, summaryBoundary);
    if (!message) {
      indexMapping[i] = [];
      continue;
    }

    const sourceContentPartOffset =
      summaryBoundary?.mode === 'positional' &&
      summaryBoundary.messageIndex === i
        ? summaryBoundary.contentIndex + 1
        : 0;

    // Q: Store the current length of messages to track where this payload message starts in the result?
    // const startIndex = messages.length;
    if (typeof message.content === 'string') {
      message = {
        ...message,
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: message.content },
        ],
      };
    } else if (Array.isArray(message.content) && message.content.length === 0) {
      indexMapping[i] = [];
      continue;
    }

    if (message.role !== 'assistant') {
      /**
       * Strict providers (Anthropic, Bedrock, Maritaca) reject a user message
       * whose effective content is empty — and with it the entire request.
       * History replay of a promptless send reaches here in exactly that
       * shape once its attachments are no longer re-sent. Dropping the turn
       * is not safe either: adjacent messages would collapse into consecutive
       * same-role turns those providers also reject. Keep the turn and give it
       * the smallest honest stand-in. Vision turns keep their image parts —
       * `userContentHasNoVisibleText` only matches text-only-empty content.
       */
      if (
        message.role === 'user' &&
        userContentHasNoVisibleText(message.content)
      ) {
        message = {
          ...message,
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: EMPTY_USER_MESSAGE_PLACEHOLDER,
            },
          ],
        };
      }
      const formattedMessage = formatMessage({
        message: message as MessageInput,
        langChain: true,
      }) as
        | RoleBearingMessage<HumanMessage>
        | RoleBearingMessage<AIMessage>
        | RoleBearingMessage<SystemMessage>;
      let attribution: ProviderMessageAttribution = 'synthetic';
      if (formattedMessage.role === 'user') {
        attribution = 'user';
      } else if (formattedMessage.role === 'assistant') {
        attribution = 'model';
      }
      const provenanceParts = createProviderContentProvenanceParts(
        message.content as MessageContentComplex[],
        sourceContentPartOffset,
        sourceMessageId,
        attribution
      );
      stampSourceMessageIdentity(
        formattedMessage,
        sourceMessageId,
        0,
        attribution,
        undefined,
        provenanceParts
      );
      flushSteerAnchor(formattedMessage);
      emitFormattedMessage(formattedMessage);

      // Update the index mapping for this message
      indexMapping[i] = [messages.length - 1];
      continue;
    }

    // For assistant messages, track the starting index before processing
    const startMessageIndex = messages.length;

    /**
     * If tools set is provided, process tool_calls:
     * - Keep valid tool_calls (tools in the set, SDK-managed, or dynamically discovered)
     * - Omit invalid tool_calls without exposing their output as assistant-authored text
     * - Dynamically expand the set when tool_search results are encountered
     */
    let processedMessage = message;
    let processedSourceContentPartIndices:
      | SourceContentPartIndices[]
      | undefined;
    const processedToolSourceContentPartIndices =
      collectTrustedToolResultSourceContentPartIndices(
        getBoundedProviderPairingArrayProperty(message, 'content'),
        sourceContentPartOffset
      );
    let pendingSkillNames: Set<string> | undefined;
    if (discoveredTools) {
      const content = message.content;
      if (content != null && Array.isArray(content)) {
        const filteredContent: typeof content = [];
        const filteredSourceContentPartIndices: SourceContentPartIndices[] = [];
        const invalidToolCallIds = new Set<string>();

        for (let partIndex = 0; partIndex < content.length; partIndex++) {
          const part = content[partIndex] as
            | MessageContentComplex
            | null
            | undefined;
          const partSourceContentIndices = sourceContentPartOffset + partIndex;
          if (part == null || typeof part !== 'object') {
            continue;
          }
          if (part.type !== ContentTypes.TOOL_CALL) {
            filteredContent.push(part);
            filteredSourceContentPartIndices.push(partSourceContentIndices);
            continue;
          }

          /** Skip malformed tool_call entries */
          if (
            part.tool_call == null ||
            part.tool_call.name == null ||
            part.tool_call.name === ''
          ) {
            if (
              typeof part.tool_call?.id === 'string' &&
              part.tool_call.id !== ''
            ) {
              invalidToolCallIds.add(part.tool_call.id);
            }
            continue;
          }

          const toolName = part.tool_call.name;

          /**
           * If this is a tool_search result with output, extract discovered tool names
           * and add them to the discoveredTools set for subsequent validation.
           */
          if (
            toolName === Constants.TOOL_SEARCH &&
            typeof part.tool_call.output === 'string' &&
            part.tool_call.output !== ''
          ) {
            const extracted = extractToolNamesFromSearchOutput(
              part.tool_call.output
            );
            for (const name of extracted) {
              discoveredTools.add(name);
            }
          }

          if (discoveredTools.has(toolName) || isSdkManagedToolName(toolName)) {
            filteredContent.push(part);
            filteredSourceContentPartIndices.push(partSourceContentIndices);
            if (
              toolName === Constants.SKILL_TOOL &&
              skills?.size != null &&
              skills.size > 0
            ) {
              const skillName = extractSkillName(part.tool_call.args) ?? '';
              if (skillName) {
                (pendingSkillNames ??= new Set()).add(skillName);
              }
            }
          } else if (
            typeof part.tool_call.id === 'string' &&
            part.tool_call.id !== ''
          ) {
            invalidToolCallIds.add(part.tool_call.id);
          }
        }

        /** Remove tool_call_ids references to invalid tools from text parts */
        if (invalidToolCallIds.size > 0) {
          for (const part of filteredContent) {
            if (
              part.type === ContentTypes.TEXT &&
              Array.isArray(part.tool_call_ids)
            ) {
              part.tool_call_ids = part.tool_call_ids.filter(
                (id: string) => !invalidToolCallIds.has(id)
              );
              if (part.tool_call_ids.length === 0) {
                delete part.tool_call_ids;
              }
            }
          }
        }

        /** Use filtered content if we made any changes */
        if (filteredContent.length !== content.length) {
          processedMessage = { ...message, content: filteredContent };
          processedSourceContentPartIndices = filteredSourceContentPartIndices;
        }
      }
    }

    /** When tools filtering is off, still detect skill tool_calls for body reconstruction */
    if (!discoveredTools && skills?.size != null && skills.size > 0) {
      const content = processedMessage.content;
      if (Array.isArray(content)) {
        for (const part of content as Array<
          MessageContentComplex | null | undefined
        >) {
          if (
            part == null ||
            typeof part !== 'object' ||
            part.type !== ContentTypes.TOOL_CALL ||
            part.tool_call?.name !== Constants.SKILL_TOOL
          ) {
            continue;
          }
          const skillName = extractSkillName(part.tool_call.args) ?? '';
          if (skillName) {
            (pendingSkillNames ??= new Set()).add(skillName);
          }
        }
      }
    }

    const formattedMessages = formatAssistantMessage(processedMessage, {
      compactionSemanticIndex: compactionSemanticIndexCollector,
      intentToolNames: options?.compactionSemanticIndex?.intentToolNames,
      preserveUnpairedServerToolUses: i === payload.length - 1,
      preserveReasoningContent:
        options?.preserveReasoningContent ??
        options?.provider === Providers.DEEPSEEK,
      provider: options?.provider,
      retainedSourceContentEnd:
        sourceContentPartOffset +
        (Array.isArray(message.content) ? message.content.length : 0),
      sourceMessageId,
      sourceContentPartOffset,
      sourceContentPartIndices: processedSourceContentPartIndices,
      toolSourceContentPartIndices: processedToolSourceContentPartIndices,
    });
    /**
     * A steer that ends an assistant message leaves the replay on a
     * `HumanMessage`. The next payload message is itself a user turn, so the
     * sequence would reach the provider as two adjacent user turns — rejected
     * by strict-alternation providers. Anchor it with a placeholder assistant
     * turn.
     *
     * The placeholder must be NON-EMPTY. A string-content assistant message
     * with no tool calls passes through `_convertMessagesToAnthropicPayload`
     * verbatim — the empty-text repair there only covers array content and
     * tool-call turns — so an empty anchor would reach Anthropic as
     * `{role: 'assistant', content: ''}` and trade one invalid sequence for
     * another. Same single-underscore convention the Anthropic converter
     * already uses when it has to synthesize a non-empty block.
     *
     * Deferred rather than decided by lookahead. `i < payload.length - 1` only
     * proves a later ENTRY exists, not that it EMITS: entries with empty
     * content, and entries dropped by `applySummaryBoundary`, are skipped
     * silently. A trailing steer followed only by those would get the anchor
     * as the FINAL turn — an assistant prefill with no request after it, which
     * the model may simply never answer. So the intent is recorded and flushed
     * only when a message actually follows.
     *
     * Pushed AFTER source identity stamping above, deliberately. The anchor is
     * synthetic rather than derived from the persisted assistant entry, so it
     * must not claim that entry's source metadata. Left unstamped, it reaches
     * the reducer with a null id and is assigned a fresh one.
     * `endsWithSteerMessage` reads only `additional_kwargs.source`, so the
     * deferral cannot change which messages get anchored.
     */
    /**
     * Guarded on emission: an assistant entry whose blocks all filtered away
     * emits nothing, and flushing for it would strand the anchor as the final
     * turn — the pending flag stays set for whichever entry emits next.
     */
    if (formattedMessages.length > 0) {
      flushSteerAnchor(formattedMessages[0]);
    }
    for (const formattedMessage of formattedMessages) {
      emitFormattedMessage(formattedMessage);
    }
    if (endsWithSteerMessage(formattedMessages)) {
      pendingSteerAnchor = true;
    }

    // Capture index range BEFORE skill body injection so injected
    // HumanMessages are excluded from the assistant's token distribution.
    const endMessageIndex = messages.length;

    if (pendingSkillNames?.size != null && pendingSkillNames.size > 0) {
      const skipSkillBodyNames = options?.skipSkillBodyNames;
      for (const skillName of pendingSkillNames) {
        if (skipSkillBodyNames != null && skipSkillBodyNames.has(skillName)) {
          continue;
        }
        const body = skills?.get(skillName) ?? '';
        if (body) {
          const skillMessage = withMessageRole(
            new HumanMessage({
              content: body,
              additional_kwargs: {
                role: 'user',
                isMeta: true,
                source: 'skill',
                skillName,
              },
            }),
            'user'
          );
          stampSourceMessageIdentity(skillMessage, undefined, 0, 'synthetic');
          emitFormattedMessage(skillMessage);
        }
      }
    }

    const resultIndices = [];
    for (let j = startMessageIndex; j < endMessageIndex; j++) {
      resultIndices.push(j);
    }
    indexMapping[i] = resultIndices;
  }

  if (indexTokenCountMap) {
    for (
      let originalIndex = 0;
      originalIndex < payload.length;
      originalIndex++
    ) {
      const resultIndices = indexMapping[originalIndex] || [];
      let tokenCount = indexTokenCountMap[originalIndex];

      if (tokenCount === undefined) {
        continue;
      }

      /**
       * Coverage mode deliberately leaves the count alone, even though the entry
       * holding the block is charged for summary text that `formatAssistantMessage`
       * filters out and `summary.tokenCount` accounts separately.
       *
       * Discounting it needs the summary's cost in the same units as
       * `indexTokenCountMap`, and that figure is not obtainable here: this
       * function receives no tokenizer, and a count recorded at write time is in
       * the writing run's units — `Run.create` derives its counter from the model
       * in play, and a consumer may supply its own — so a conversation continued
       * on a different model would subtract across encodings. Attempts to proxy
       * it (character ratios, provider identity) all under-count some shape,
       * which risks an over-context request; over-counting merely prunes early.
       * Fixing it properly means passing the reader a tokenizer, which is a
       * consumer-facing change and out of scope here.
       */
      if (
        summaryBoundary?.mode === 'positional' &&
        originalIndex === summaryBoundary.messageIndex &&
        Array.isArray(payload[originalIndex].content)
      ) {
        const content = payload[originalIndex]
          .content as MessageContentComplex[];
        const { contentIndex } = summaryBoundary;
        if (contentIndex >= 0 && contentIndex < content.length - 1) {
          let totalCharLen = 0;
          let remainingCharLen = 0;
          /**
           * The ratio applies only when *every* part of the entry is one whose
           * token cost tracks its character length. A single ineligible part
           * cancels the discount, whichever side of the boundary it sits on.
           *
           * Both sides can break it, in opposite directions. A retained image has
           * its fixed cost scaled away, collapsing the entry. A removed base64
           * payload inflates the denominator — serializing to a huge length while
           * the counter charges a fixed estimate — dragging retained text below
           * its real cost. Either way the request can exceed the window.
           *
           * Telling a text-bearing tool payload from a media-bearing one means
           * recursing into arbitrary nested output, which has already missed a
           * level twice here. Cancelling instead keeps the original count: an
           * over-count that prunes early rather than overflowing. Entries of
           * plain text and reasoning — the common shape — still proportion.
           */
          let everyRetainedPartMeasurable = true;
          for (let p = 0; p < content.length; p++) {
            const part = content[p];
            const retained = p > contentIndex;

            if (isDroppedByFormatting(part)) {
              /** Removed summary text is real removed content: it is read as
               *  plain text and belongs in the denominator. */
              if (!retained && part.type === ContentTypes.SUMMARY) {
                totalCharLen += contentPartCharLength(part);
              }
              continue;
            }

            const charLen = contentPartCharLength(part);
            if (!isCharRatioEligible(part) || (retained && charLen === 0)) {
              everyRetainedPartMeasurable = false;
              break;
            }
            totalCharLen += charLen;
            if (retained) {
              remainingCharLen += charLen;
            }
          }
          if (totalCharLen > 0 && everyRetainedPartMeasurable) {
            const original = tokenCount;
            tokenCount = Math.max(
              1,
              Math.round(tokenCount * (remainingCharLen / totalCharLen))
            );
            boundaryTokenAdjustment = {
              original,
              adjusted: tokenCount,
              remainingChars: remainingCharLen,
              totalChars: totalCharLen,
            };
          }
        }
      }

      const msgCount = resultIndices.length;
      if (msgCount === 1) {
        updatedIndexTokenCountMap[resultIndices[0]] = tokenCount;
        continue;
      }

      if (msgCount < 2) {
        continue;
      }

      let totalLength = 0;
      const lastIdx = msgCount - 1;
      const lengths = new Array<number>(msgCount);
      for (let k = 0; k < msgCount; k++) {
        const msg = messages[resultIndices[k]];
        const { content } = msg;
        let len = 0;
        if (typeof content === 'string') {
          len = content.length;
        } else if (Array.isArray(content)) {
          for (const part of content as Array<
            Record<string, unknown> | string | undefined
          >) {
            if (typeof part === 'string') {
              len += part.length;
            } else if (part != null && typeof part === 'object') {
              const val = part.text ?? part.content;
              if (typeof val === 'string') {
                len += val.length;
              }
            }
          }
        }
        const toolCalls = (msg as AIMessage).tool_calls;
        if (Array.isArray(toolCalls)) {
          for (const tc of toolCalls as Array<Record<string, unknown>>) {
            if (typeof tc.name === 'string') {
              len += tc.name.length;
            }
            const { args } = tc;
            if (typeof args === 'string') {
              len += args.length;
            } else if (args != null) {
              const measured = serializeStructuredValueBounded(
                args,
                0
              ).originalChars;
              len +=
                measured === Number.MAX_SAFE_INTEGER
                  ? HARD_MAX_TOOL_RESULT_CHARS
                  : Math.min(measured, HARD_MAX_TOOL_RESULT_CHARS);
            }
          }
        }
        lengths[k] = len;
        totalLength += len;
      }

      if (totalLength === 0) {
        const countPerMessage = Math.floor(tokenCount / msgCount);
        for (let k = 0; k < lastIdx; k++) {
          updatedIndexTokenCountMap[resultIndices[k]] = countPerMessage;
        }
        updatedIndexTokenCountMap[resultIndices[lastIdx]] =
          tokenCount - countPerMessage * lastIdx;
      } else {
        let distributed = 0;
        for (let k = 0; k < lastIdx; k++) {
          const share = Math.floor((lengths[k] / totalLength) * tokenCount);
          updatedIndexTokenCountMap[resultIndices[k]] = share;
          distributed += share;
        }
        updatedIndexTokenCountMap[resultIndices[lastIdx]] =
          tokenCount - distributed;
      }
    }
  }

  const compactionSemanticIndexSnapshot =
    finalizeDerivedCompactionSemanticIndexSnapshot(
      compactionSemanticIndexCollector
    );
  return {
    messages,
    indexTokenCountMap: indexTokenCountMap
      ? updatedIndexTokenCountMap
      : undefined,
    summary: summaryBoundary
      ? { text: summaryBoundary.text, tokenCount: summaryBoundary.tokenCount }
      : undefined,
    boundaryTokenAdjustment,
    compactionSemanticIndex: compactionSemanticIndexSnapshot?.entries,
    compactionSemanticIndexSnapshot,
  };
};

/**
 * Adds a value at key 0 for system messages and shifts all key indices by one in an indexTokenCountMap.
 * This is useful when adding a system message at the beginning of a conversation.
 *
 * @param indexTokenCountMap - The original map of message indices to token counts
 * @param instructionsTokenCount - The token count for the system message to add at index 0
 * @returns A new map with the system message at index 0 and all other indices shifted by 1
 */
export function shiftIndexTokenCountMap(
  indexTokenCountMap: Record<number, number>,
  instructionsTokenCount: number
): Record<number, number> {
  // Create a new map to avoid modifying the original
  const shiftedMap: Record<number, number> = {};
  shiftedMap[0] = instructionsTokenCount;

  // Shift all existing indices by 1
  for (const [indexStr, tokenCount] of Object.entries(indexTokenCountMap)) {
    const index = Number(indexStr);
    shiftedMap[index + 1] = tokenCount;
  }

  return shiftedMap;
}

/** Checks whether a BaseMessage is a tool-role message. */
const isToolMessage = (m: BaseMessage): boolean =>
  m instanceof ToolMessage || ('role' in m && (m as any).role === 'tool');

const PORTABLE_FOLDED_MEDIA_TYPES = new Set(['image', 'image_url']);
const MAX_FOLDED_BLOCK_CHARS = 8_000;
const MAX_FOLDED_CONTEXT_CHARS = HARD_MAX_TOOL_RESULT_CHARS;
const MAX_FOLDED_CONTEXT_WORK = 100_000;
const FOLDED_CONTEXT_TRUNCATION_NOTICE =
  '… [additional folded context omitted]';
const syntheticProviderContextMessages = new WeakSet<BaseMessage>();

type FoldContextBudget = {
  remainingChars: number;
  remainingWork: number;
  truncated: boolean;
};

function createFoldContextBudget(): FoldContextBudget {
  return {
    remainingChars: MAX_FOLDED_CONTEXT_CHARS,
    remainingWork: MAX_FOLDED_CONTEXT_WORK,
    truncated: false,
  };
}

function readFoldedDataProperty(value: unknown, key: string): unknown {
  if (value == null || typeof value !== 'object') {
    return undefined;
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor != null && 'value' in descriptor
      ? descriptor.value
      : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Adds one logical line without first concatenating caller-controlled strings.
 * The shared budget covers every synthetic message emitted by one transform,
 * so many individually bounded blocks cannot build an unbounded aggregate.
 */
function appendFoldedLine(
  textChunks: string[],
  budget: FoldContextBudget,
  pieces: readonly string[]
): boolean {
  if (budget.remainingChars <= 0) {
    budget.truncated = true;
    return false;
  }

  const separatorChars = textChunks.length > 0 ? 1 : 0;
  const available = budget.remainingChars - separatorChars;
  if (available <= 0) {
    budget.remainingChars = 0;
    budget.truncated = true;
    return false;
  }

  let totalChars = 0;
  for (const piece of pieces) {
    totalChars = Math.min(Number.MAX_SAFE_INTEGER, totalChars + piece.length);
  }
  if (totalChars <= available) {
    const line = pieces.join('');
    textChunks.push(line);
    budget.remainingChars -= separatorChars + line.length;
    return true;
  }

  const notice = FOLDED_CONTEXT_TRUNCATION_NOTICE.slice(0, available);
  let remainingHeadChars = Math.max(0, available - notice.length);
  const boundedPieces: string[] = [];
  for (const piece of pieces) {
    if (remainingHeadChars <= 0) {
      break;
    }
    const retained = piece.slice(0, remainingHeadChars);
    boundedPieces.push(retained);
    remainingHeadChars -= retained.length;
  }
  boundedPieces.push(notice);
  textChunks.push(boundedPieces.join(''));
  budget.remainingChars = 0;
  budget.truncated = true;
  return false;
}

function consumeFoldedWork(
  textChunks: string[],
  budget: FoldContextBudget
): boolean {
  if (budget.remainingChars <= 0) {
    return false;
  }
  if (budget.remainingWork-- > 0) {
    return true;
  }
  appendFoldedLine(textChunks, budget, [FOLDED_CONTEXT_TRUNCATION_NOTICE]);
  budget.remainingChars = 0;
  budget.truncated = true;
  return false;
}

function serializeFoldedValue(value: unknown): string {
  const compacted = compactToolContent(value, MAX_FOLDED_BLOCK_CHARS).content;
  return typeof compacted === 'string'
    ? compacted
    : serializeStructuredValueBounded(compacted, MAX_FOLDED_BLOCK_CHARS)
      .content;
}

interface FoldedSourceMessage {
  readonly message: BaseMessage;
  /** Provider-content positions retained from an array message. Omitted when
   * string content or tool-call metadata contributed as a whole. */
  readonly retainedContentPartIndices?: ReadonlySet<number>;
  readonly mappingAmbiguous?: boolean;
  readonly additionalAttributions?: readonly ProviderMessageAttribution[];
}

function getFoldedSourceAttribution(
  message: BaseMessage
): ProviderMessageAttribution {
  const messageType = message.getType();
  if (messageType === 'ai') {
    return 'model';
  }
  if (messageType === 'tool') {
    return 'tool';
  }
  if (messageType === 'system') {
    return 'synthetic';
  }
  return 'user';
}

function getSyntheticProviderContextProvenanceParts(
  sourceMessages: readonly FoldedSourceMessage[]
): ProviderMessageProvenancePart[] | null {
  /** Fold labels are generated context, while retained source bytes keep their
   * original attribution so downstream policy can still route them exactly. */
  const parts: ProviderMessageProvenancePart[] = [{ attribution: 'synthetic' }];
  for (const source of sourceMessages) {
    const {
      message: sourceMessage,
      retainedContentPartIndices,
      mappingAmbiguous,
    } = source;
    const provenanceState = inspectProviderMessageProvenance(sourceMessage);
    const sourceMessageIdsState =
      inspectProviderSourceMessageIds(sourceMessage);
    if (
      provenanceState.status === 'invalid' ||
      sourceMessageIdsState.status === 'invalid'
    ) {
      return null;
    }
    const explicit =
      provenanceState.status === 'valid'
        ? provenanceState.provenance
        : undefined;
    const sourceMessageIds =
      sourceMessageIdsState.status === 'valid'
        ? sourceMessageIdsState.sourceMessageIds
        : [];
    const fallbackAttribution = getFoldedSourceAttribution(sourceMessage);
    const sourcePartStart = parts.length;
    const retainedSourceIds = new Set<string>();
    const contentLength = Array.isArray(sourceMessage.content)
      ? sourceMessage.content.length
      : undefined;
    const mapsOneToOne =
      mappingAmbiguous !== true &&
      retainedContentPartIndices != null &&
      contentLength != null &&
      explicit != null &&
      hasBijectiveProviderContentPartMapping(explicit.parts, contentLength);
    const retainedAllContentParts =
      mappingAmbiguous !== true &&
      retainedContentPartIndices != null &&
      contentLength != null &&
      retainedContentPartIndices.size === contentLength;
    const retainUnindexedSourceIds =
      (mappingAmbiguous !== true && retainedContentPartIndices == null) ||
      retainedAllContentParts ||
      sourceMessageIds.length <= 1;
    for (const part of explicit?.parts ?? []) {
      let sourceContentPartIndices = part.sourceContentPartIndices;
      if (mappingAmbiguous === true && sourceContentPartIndices != null) {
        sourceContentPartIndices = undefined;
      }
      if (
        retainedContentPartIndices != null &&
        sourceContentPartIndices != null &&
        !retainedAllContentParts
      ) {
        if (!mapsOneToOne) {
          sourceContentPartIndices = undefined;
        } else {
          const retainedSourceContentPartIndices: number[] = [];
          for (const sourceContentPartIndex of sourceContentPartIndices) {
            if (retainedContentPartIndices.has(sourceContentPartIndex)) {
              retainedSourceContentPartIndices.push(sourceContentPartIndex);
            }
          }
          if (retainedSourceContentPartIndices.length === 0) {
            continue;
          }
          sourceContentPartIndices = retainedSourceContentPartIndices;
        }
      }
      const sourceMessageId =
        sourceContentPartIndices != null || retainUnindexedSourceIds
          ? part.sourceMessageId
          : undefined;
      parts.push({
        attribution: part.attribution,
        ...(sourceMessageId != null && {
          sourceMessageId,
        }),
        ...(sourceContentPartIndices != null && {
          sourceContentPartIndices,
        }),
      });
      if (sourceMessageId != null) {
        retainedSourceIds.add(sourceMessageId);
      }
    }
    for (const sourceMessageId of sourceMessageIds) {
      if (!retainUnindexedSourceIds) {
        break;
      }
      if (!retainedSourceIds.has(sourceMessageId)) {
        parts.push({ attribution: fallbackAttribution, sourceMessageId });
      }
    }
    if (parts.length === sourcePartStart) {
      parts.push({ attribution: fallbackAttribution });
    }
    for (const attribution of source.additionalAttributions ?? []) {
      const sourceMessageId =
        sourceMessageIds.length === 1 ? sourceMessageIds[0] : undefined;
      const retainedSourceId =
        retainedSourceIds.size === 1
          ? retainedSourceIds.values().next().value
          : undefined;
      parts.push({
        attribution,
        ...((sourceMessageId ?? retainedSourceId) != null && {
          sourceMessageId: sourceMessageId ?? retainedSourceId,
        }),
      });
    }
  }
  return mergeAdjacentProviderProvenanceParts(parts);
}

function markSyntheticProviderContext<T extends BaseMessage>(
  message: T,
  sourceMessages: readonly FoldedSourceMessage[] = []
): T {
  syntheticProviderContextMessages.add(message);
  const parts = getSyntheticProviderContextProvenanceParts(sourceMessages);
  if (parts == null) {
    setInvalidProviderMessageProvenance(message);
    return message;
  }
  setProviderMessageProvenance(
    message,
    parts.length > 0 ? parts : [{ attribution: 'synthetic' }]
  );
  return message;
}

function appendSyntheticProviderContextMessage(
  result: BaseMessage[],
  parts: MessageContentComplex[],
  sourceMessages: readonly FoldedSourceMessage[] = []
): boolean {
  if (parts.length === 0) {
    return false;
  }
  result.push(
    markSyntheticProviderContext(
      withMessageRole(
        new HumanMessage({ content: toLangChainContent(parts) }),
        'user'
      ),
      sourceMessages
    )
  );
  return true;
}

/**
 * Identifies provider-context placeholders created by this module without
 * trusting user-controlled content prefixes or leaking marker metadata onto
 * the provider wire.
 */
export function isSyntheticProviderContextMessage(
  message: BaseMessage
): boolean {
  return syntheticProviderContextMessages.has(message);
}

/** Compacts a folded provider-context message without retaining source claims
 * for bytes whose mapping cannot survive the lossy compaction. The unsourced
 * user/tool parts intentionally make security consumers inspect the exact
 * compacted wire under both external trust domains. */
export function compactSyntheticProviderContextMessage(
  message: HumanMessage,
  maxChars: number
): HumanMessage {
  const compactedContent = compactToolContent(message.content, maxChars);
  if (!compactedContent.changed) {
    return message;
  }
  const compacted = new HumanMessage({
    content: compactedContent.content,
    id: message.id,
    name: message.name,
    additional_kwargs: { ...message.additional_kwargs },
    response_metadata: message.response_metadata,
  });
  syntheticProviderContextMessages.add(compacted);
  setProviderMessageProvenance(compacted, [
    { attribution: 'synthetic' },
    { attribution: 'user' },
    { attribution: 'tool' },
  ]);
  return compacted;
}

/** Flushes accumulated text chunks into `parts` as a single text block. */
function flushTextChunks(
  textChunks: string[],
  parts: MessageContentComplex[]
): void {
  if (textChunks.length === 0) {
    return;
  }
  parts.push({
    type: ContentTypes.TEXT,
    text: textChunks.join('\n'),
  } as MessageContentComplex);
  textChunks.length = 0;
}

function getAuthoritativeResponsesToolOutput(
  message: BaseMessage,
  preparation: ToolHistoryPreparation
): OrderedToolHistoryProjection | undefined {
  const projection = preparation.get(message);
  return projection?.hasToolContent === true ? projection : undefined;
}

function appendOrderedResponsesOutput(
  projection: OrderedToolHistoryProjection,
  textChunks: string[],
  parts: MessageContentComplex[],
  budget: FoldContextBudget,
  omitText = false
): { contributed: boolean; complete: boolean; hasTool: boolean } {
  let contributed = false;
  let hasTool = false;
  for (const contribution of projection.contributions) {
    if (!consumeFoldedWork(textChunks, budget)) {
      break;
    }
    const remainingCharsBefore = budget.remainingChars;
    if (contribution.kind === 'image') {
      const chars = getToolContentCharLength([contribution.image]);
      if (chars > budget.remainingChars) {
        appendFoldedLine(textChunks, budget, [
          'Tool: [image omitted: folded context limit]',
        ]);
        contributed = true;
        hasTool = true;
        continue;
      }
      flushTextChunks(textChunks, parts);
      parts.push(contribution.image as MessageContentComplex);
      budget.remainingChars -= chars;
      contributed = true;
      hasTool = true;
      continue;
    }
    if (contribution.kind === 'text') {
      if (omitText) {
        continue;
      }
      appendFoldedLine(textChunks, budget, ['AI: ', contribution.text]);
    } else if (contribution.kind === 'call') {
      appendFoldedLine(textChunks, budget, [
        `AI: [tool_call] ${contribution.name}(`,
        serializeFoldedValue(contribution.arguments),
        ')',
      ]);
    } else if (contribution.providerType !== 'reasoning') {
      appendFoldedLine(textChunks, budget, [
        'AI: [server_tool_output] ',
        serializeFoldedValue(contribution.value),
      ]);
    }
    const retained = budget.remainingChars < remainingCharsBefore;
    contributed ||= retained;
    hasTool ||= retained && contribution.actor === 'tool';
  }
  if (projection.truncated) {
    const before = budget.remainingChars;
    appendFoldedLine(textChunks, budget, [FOLDED_CONTEXT_TRUNCATION_NOTICE]);
    contributed ||= budget.remainingChars < before;
  }
  return {
    contributed,
    complete: !budget.truncated && !projection.truncated,
    hasTool,
  };
}

/**
 * Appends a single message's content to the running `textChunks` / `parts`
 * accumulators. Portable image blocks are shallow-copied into `parts` so
 * binary data never becomes text tokens. Provider-specific media/resource
 * blocks are retained as bounded text so a folded cross-provider history
 * cannot contain an unsupported native block or JSON-expand without limit.
 *
 * Parsed calls are omitted only when identity, name, and complete arguments
 * establish an equivalent content representation.
 */
function appendMessageContent(
  msg: BaseMessage,
  role: string,
  textChunks: string[],
  parts: MessageContentComplex[],
  budget: FoldContextBudget,
  preparation: ToolHistoryPreparation = createToolHistoryPreparation()
): FoldedSourceMessage | undefined {
  const { content } = msg;
  const responsesOutput = getAuthoritativeResponsesToolOutput(msg, preparation);
  if (
    responsesOutput != null &&
    (responsesOutput.source.coverage === 'complete-output' ||
      typeof content === 'string' ||
      (Array.isArray(content) &&
        content.length <= MAX_FOLDED_CONTEXT_WORK &&
        content.every(
          (block) => readFoldedDataProperty(block, 'type') === 'text'
        )))
  ) {
    const orderedOutput = appendOrderedResponsesOutput(
      responsesOutput,
      textChunks,
      parts,
      budget
    );
    const toolCalls =
      responsesOutput.source.coverage === 'tool-sidecar'
        ? appendToolCalls(msg, role, textChunks, budget)
        : { contributed: false, complete: true };
    return orderedOutput.contributed || toolCalls.contributed
      ? {
        message: msg,
        mappingAmbiguous: true,
        ...(orderedOutput.hasTool && {
          additionalAttributions: ['tool'] as const,
        }),
      }
      : undefined;
  }

  if (typeof content === 'string') {
    const remainingCharsBefore = budget.remainingChars;
    let contentComplete = true;
    if (content) {
      contentComplete =
        consumeFoldedWork(textChunks, budget) &&
        appendFoldedLine(textChunks, budget, [`${role}: `, content]);
    }
    const toolCalls = appendToolCalls(msg, role, textChunks, budget);
    return budget.remainingChars < remainingCharsBefore || toolCalls.contributed
      ? {
        message: msg,
        ...((!contentComplete || !toolCalls.complete) && {
          mappingAmbiguous: true,
        }),
      }
      : undefined;
  }

  if (!Array.isArray(content)) {
    const toolCalls = appendToolCalls(msg, role, textChunks, budget);
    return toolCalls.contributed
      ? {
        message: msg,
        ...(!toolCalls.complete && { mappingAmbiguous: true }),
      }
      : undefined;
  }

  const representedToolCalls: ToolHistoryCallMirrors = new Map();
  const retainedContentPartIndices = new Set<number>();

  for (let blockIndex = 0; blockIndex < content.length; blockIndex++) {
    const block = content[blockIndex] as ExtendedMessageContent;
    if (!consumeFoldedWork(textChunks, budget)) {
      break;
    }
    const remainingCharsBefore = budget.remainingChars;
    const markBlockRetained = (): void => {
      if (budget.remainingChars < remainingCharsBefore) {
        retainedContentPartIndices.add(blockIndex);
      }
    };
    const blockTypeValue = readFoldedDataProperty(block, 'type');
    const blockType =
      typeof blockTypeValue === 'string' ? blockTypeValue : undefined;
    const blockRole =
      getProviderToolResultPartDescriptor(block) != null ? 'Tool' : role;
    const callDescriptor = getProviderToolCallPartDescriptor(block);
    if (callDescriptor != null) {
      recordToolHistoryCallMirror(representedToolCalls, block);
    }
    if (
      blockType !== 'tool_use' &&
      blockType !== 'tool_call' &&
      blockType !== 'tool_result' &&
      isAtomicToolContentBlock(block)
    ) {
      if (PORTABLE_FOLDED_MEDIA_TYPES.has(blockType ?? '')) {
        const blockChars = getToolContentCharLength([block]);
        if (blockChars > budget.remainingChars) {
          appendFoldedLine(textChunks, budget, [
            `${blockRole}: [${blockType ?? 'media'} omitted: folded context limit]`,
          ]);
          markBlockRetained();
          continue;
        }
        flushTextChunks(textChunks, parts);
        parts.push({ ...block } as MessageContentComplex);
        budget.remainingChars -= blockChars;
      } else {
        appendFoldedLine(textChunks, budget, [
          `${blockRole}: [${blockType ?? 'media'}] `,
          serializeFoldedValue(block),
        ]);
      }
      markBlockRetained();
      continue;
    }

    if (blockType === 'tool_use') {
      const name = readFoldedDataProperty(block, 'name');
      const input = readFoldedDataProperty(block, 'input');
      appendFoldedLine(textChunks, budget, [
        `${role}: [tool_use] ${typeof name === 'string' ? name : ''} `,
        serializeFoldedValue(input ?? {}),
      ]);
      markBlockRetained();
      continue;
    }

    // A `tool_call` content block appears either as the v1 standard shape
    // (`{ name, args }` at top level, which `@langchain/aws` maps to a Converse
    // toolUse) or this repo's `ToolCallContent` (`{ tool_call: { name, args,
    // output } }`, from `convertMessagesToContent` / persisted history). Handle
    // both, and emit any embedded output, so the name/args/result survive.
    if (blockType === 'tool_call') {
      const nested = readFoldedDataProperty(block, 'tool_call');
      const nestedName = readFoldedDataProperty(nested, 'name');
      const topLevelName = readFoldedDataProperty(block, 'name');
      let name = '';
      if (typeof nestedName === 'string') {
        name = nestedName;
      } else if (typeof topLevelName === 'string') {
        name = topLevelName;
      }
      const nestedArgs = readFoldedDataProperty(nested, 'args');
      const topLevelArgs = readFoldedDataProperty(block, 'args');
      const rawArgs = nestedArgs ?? topLevelArgs ?? {};
      const argsText =
        typeof rawArgs === 'string' ? rawArgs : serializeFoldedValue(rawArgs);
      appendFoldedLine(textChunks, budget, [
        `${role}: [tool_use] ${name}${argsText ? ' ' : ''}`,
        argsText,
      ]);
      const output = readFoldedDataProperty(nested, 'output');
      if (output != null && output !== '') {
        appendFoldedLine(textChunks, budget, [
          'Tool: ',
          typeof output === 'string' ? output : serializeFoldedValue(output),
        ]);
      }
      markBlockRetained();
      continue;
    }

    // A `tool_result` content block (e.g. an AIMessage(tool_call) followed by a
    // user message carrying the result). Preserve nested image blocks as-is
    // instead of JSON-stringifying them through the generic fallback.
    if (blockType === 'tool_result') {
      const inner = readFoldedDataProperty(block, 'content') as
        | ToolResultContent['content']
        | undefined;
      if (typeof inner === 'string') {
        if (inner) {
          appendFoldedLine(textChunks, budget, ['Tool: [tool_result] ', inner]);
        }
      } else if (Array.isArray(inner)) {
        for (const innerBlock of inner as Array<
          string | ExtendedMessageContent
        >) {
          if (!consumeFoldedWork(textChunks, budget)) {
            break;
          }
          if (typeof innerBlock === 'string') {
            if (innerBlock) {
              appendFoldedLine(textChunks, budget, [
                'Tool: [tool_result] ',
                innerBlock,
              ]);
            }
          } else {
            const innerTypeValue = readFoldedDataProperty(innerBlock, 'type');
            const innerType =
              typeof innerTypeValue === 'string' ? innerTypeValue : undefined;
            if (
              isAtomicToolContentBlock(innerBlock) &&
              PORTABLE_FOLDED_MEDIA_TYPES.has(innerType ?? '')
            ) {
              const blockChars = getToolContentCharLength([innerBlock]);
              if (blockChars <= budget.remainingChars) {
                flushTextChunks(textChunks, parts);
                parts.push({ ...innerBlock } as MessageContentComplex);
                budget.remainingChars -= blockChars;
              } else {
                appendFoldedLine(textChunks, budget, [
                  `Tool: [${innerType ?? 'media'} omitted: folded context limit]`,
                ]);
              }
            } else {
              const textValue = readFoldedDataProperty(innerBlock, 'text');
              const inputValue = readFoldedDataProperty(innerBlock, 'input');
              const innerText = textValue ?? inputValue;
              appendFoldedLine(textChunks, budget, [
                'Tool: [tool_result] ',
                typeof innerText === 'string' && innerText
                  ? innerText
                  : serializeFoldedValue(innerBlock),
              ]);
            }
          }
        }
      } else if (inner != null) {
        appendFoldedLine(textChunks, budget, [
          'Tool: [tool_result] ',
          serializeFoldedValue(inner),
        ]);
      }
      markBlockRetained();
      continue;
    }

    const text =
      readFoldedDataProperty(block, 'text') ??
      readFoldedDataProperty(block, 'input');
    if (typeof text === 'string' && text) {
      appendFoldedLine(textChunks, budget, [`${role}: `, text]);
      markBlockRetained();
      continue;
    }

    // Fallback: serialize unrecognized block types to preserve context
    if (blockType != null && blockType !== '') {
      appendFoldedLine(textChunks, budget, [
        `${blockRole}: [${blockType}] `,
        serializeFoldedValue(block),
      ]);
    }
    markBlockRetained();
  }

  const responsesOutputContributed =
    responsesOutput != null &&
    appendOrderedResponsesOutput(
      responsesOutput,
      textChunks,
      parts,
      budget,
      true
    ).contributed;

  const toolCalls = appendToolCalls(
    msg,
    role,
    textChunks,
    budget,
    representedToolCalls
  );
  if (toolCalls.contributed || responsesOutputContributed) {
    return {
      message: msg,
      ...((responsesOutputContributed || !toolCalls.complete) && {
        mappingAmbiguous: true,
      }),
      ...(responsesOutputContributed && {
        additionalAttributions: ['tool'] as const,
      }),
    };
  }
  return retainedContentPartIndices.size > 0
    ? { message: msg, retainedContentPartIndices }
    : undefined;
}

function appendToolCalls(
  msg: BaseMessage,
  role: string,
  textChunks: string[],
  budget: FoldContextBudget,
  representedToolCalls?: ToolHistoryCallMirrors
): { contributed: boolean; complete: boolean } {
  if (role !== 'AI') {
    return { contributed: false, complete: true };
  }
  const remainingCharsBefore = budget.remainingChars;
  let complete = true;
  const aiMsg = msg as AIMessage;
  if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
    const rawToolCalls = aiMsg.additional_kwargs.tool_calls;
    if (Array.isArray(rawToolCalls)) {
      if (rawToolCalls.length !== aiMsg.tool_calls.length) {
        complete = false;
      } else {
        for (let index = 0; index < rawToolCalls.length; index++) {
          const rawToolCall = rawToolCalls[index];
          const parsedToolCall = aiMsg.tool_calls[index];
          const fn = readFoldedDataProperty(rawToolCall, 'function');
          const rawId = readFoldedDataProperty(rawToolCall, 'id');
          const rawName = readFoldedDataProperty(fn, 'name');
          const parsedId = readFoldedDataProperty(parsedToolCall, 'id');
          const parsedName = readFoldedDataProperty(parsedToolCall, 'name');
          if (fn == null || rawId !== parsedId || rawName !== parsedName) {
            complete = false;
            break;
          }
        }
      }
    }
    for (const tc of aiMsg.tool_calls) {
      if (!consumeFoldedWork(textChunks, budget)) {
        complete = false;
        break;
      }
      const name = readFoldedDataProperty(tc, 'name');
      const args = readFoldedDataProperty(tc, 'args');
      if (
        isToolHistoryCallMirror(
          representedToolCalls,
          readFoldedDataProperty(tc, 'id'),
          name,
          args
        )
      ) {
        continue;
      }
      appendFoldedLine(textChunks, budget, [
        `AI: [tool_call] ${typeof name === 'string' ? name : ''}(`,
        serializeFoldedValue(args),
        ')',
      ]);
    }
    return {
      contributed: budget.remainingChars < remainingCharsBefore,
      complete,
    };
  }
  // Fall back to raw provider tool calls kept only in additional_kwargs.
  const rawToolCalls = aiMsg.additional_kwargs.tool_calls;
  if (!Array.isArray(rawToolCalls)) {
    return { contributed: false, complete: true };
  }
  for (const tc of rawToolCalls) {
    if (!consumeFoldedWork(textChunks, budget)) {
      complete = false;
      break;
    }
    const fn = readFoldedDataProperty(tc, 'function');
    if (fn == null) {
      complete = false;
      continue;
    }
    const name = readFoldedDataProperty(fn, 'name');
    const args = readFoldedDataProperty(fn, 'arguments');
    appendFoldedLine(textChunks, budget, [
      `AI: [tool_call] ${typeof name === 'string' ? name : ''}(`,
      serializeFoldedValue(typeof args === 'string' ? args : ''),
      ')',
    ]);
  }
  return {
    contributed: budget.remainingChars < remainingCharsBefore,
    complete,
  };
}

/**
 * Ensures compatibility when switching from a non-thinking agent to a thinking-enabled agent.
 * Converts AI messages with tool calls (that lack thinking/reasoning blocks) into buffer strings,
 * avoiding the thinking block signature requirement.
 *
 * Recognizes the following as valid thinking/reasoning blocks:
 * - ContentTypes.THINKING (Anthropic)
 * - ContentTypes.REASONING_CONTENT (Bedrock)
 * - ContentTypes.REASONING (VertexAI / Google)
 * - 'redacted_thinking'
 *
 * @param messages - Array of messages to process
 * @param provider - The provider being used (unused but kept for future compatibility)
 * @param config - Optional RunnableConfig for structured agent logging
 * @param runStartIndex - Index in `messages` where the CURRENT run's own
 *   appended AI/Tool messages begin (i.e. anything at this index or later
 *   was just produced by this run's own iterations, not historical
 *   context). When provided, AI messages at or after this index are
 *   never converted to `[Previous agent context]` placeholders — Claude
 *   can validly skip a thinking block before a tool_use (cf. PR #116),
 *   so the agent's own in-run iterations must not be misclassified as
 *   foreign history. Without the signal the function falls back to its
 *   prior heuristic (`chainHasThinkingBlock`), preserving backward
 *   compatibility for callers that don't yet pass the boundary.
 * @returns The messages array with tool sequences converted to buffer strings if necessary
 */
export function ensureThinkingBlockInMessages(
  messages: BaseMessage[],
  _provider: ProviderName,
  config?: RunnableConfig,
  runStartIndex?: number,
  preparation: ToolHistoryPreparation = createToolHistoryPreparation()
): BaseMessage[] {
  if (messages.length === 0) {
    return messages;
  }

  // Find the last HumanMessage. Only the trailing sequence after it needs
  // validation — earlier messages are history already accepted by the provider.
  let lastHumanIndex = -1;
  for (let k = messages.length - 1; k >= 0; k--) {
    const m = messages[k];
    if (
      m instanceof HumanMessage ||
      ('role' in m && (m as any).role === 'user')
    ) {
      lastHumanIndex = k;
      break;
    }
  }

  if (lastHumanIndex === messages.length - 1) {
    return messages;
  }

  const result: BaseMessage[] =
    lastHumanIndex >= 0 ? messages.slice(0, lastHumanIndex + 1) : [];
  const foldBudget = createFoldContextBudget();
  let i = lastHumanIndex + 1;

  while (i < messages.length) {
    const msg = messages[i];
    /** Detect AI messages by instanceof OR by role, in case cache-control cloning
     produced a plain object that lost the LangChain prototype. */
    const isAI =
      msg instanceof AIMessage ||
      msg instanceof AIMessageChunk ||
      ('role' in msg && (msg as any).role === 'assistant');

    if (!isAI) {
      result.push(msg);
      i++;
      continue;
    }

    const aiMsg = msg as AIMessage | AIMessageChunk;
    const hasToolCalls = aiMsg.tool_calls && aiMsg.tool_calls.length > 0;
    const contentIsArray = Array.isArray(aiMsg.content);

    // Check if the message has tool calls or tool_use content
    let hasToolUse = hasToolCalls ?? false;
    let hasThinkingBlock = false;

    if (contentIsArray && aiMsg.content.length > 0) {
      for (const c of aiMsg.content as ExtendedMessageContent[]) {
        if (typeof c !== 'object') {
          continue;
        }
        const type = readFoldedDataProperty(c, 'type');
        if (type === 'tool_use') {
          hasToolUse = true;
        } else if (isReasoningContentBlock({ type })) {
          hasThinkingBlock = true;
        }
        if (hasToolUse && hasThinkingBlock) {
          break;
        }
      }
    }

    // Bedrock also stores reasoning in additional_kwargs (may not be in content array)
    if (
      !hasThinkingBlock &&
      aiMsg.additional_kwargs.reasoning_content != null
    ) {
      hasThinkingBlock = true;
    }

    // If message has tool use but no thinking block, check whether this is a
    // continuation of a thinking-enabled agent's chain before converting.
    // Bedrock reasoning models can produce multiple AI→Tool rounds after an
    // initial reasoning response: the first AI message has reasoning_content,
    // but follow-ups have content: "" with only tool_calls. These are the
    // same agent's turn and must NOT be converted to HumanMessages.
    if (hasToolUse && !hasThinkingBlock) {
      // Current-run boundary check: anything at or after `runStartIndex`
      // is the current run's own work — preserve it. Claude is allowed
      // to skip a thinking block before a tool_use (cf. PR #116 in the
      // agents repo), so the agent's own first-iteration AI message can
      // legitimately have tool_calls without reasoning. Converting it to
      // a `[Previous agent context]` placeholder pollutes the next
      // iteration's prompt — the LLM sees the placeholder, treats it as
      // suspicious injected content, ignores its own real prior tool
      // result, and re-runs the tool to verify (which then often fails
      // because subsequent calls land in fresh sandboxes without the
      // file). Skip the conversion when we know this is in-run.
      if (runStartIndex !== undefined && i >= runStartIndex) {
        result.push(msg);
        i++;
        continue;
      }

      // Walk backwards — if an earlier AI message in the same chain (before
      // the nearest HumanMessage) has a thinking/reasoning block, this is a
      // continuation of a thinking-enabled turn, not a non-thinking handoff.
      if (chainHasThinkingBlock(messages, i)) {
        result.push(msg);
        i++;
        continue;
      }

      // Build structured content in a single pass over the AI + following
      // ToolMessages — preserves image blocks as-is to avoid serializing
      // binary data as text (which caused 174× token amplification).
      const parts: MessageContentComplex[] = [];
      const textChunks: string[] = [];
      const foldedSourceMessages: FoldedSourceMessage[] = [];
      appendFoldedLine(textChunks, foldBudget, ['[Previous agent context]']);

      const aiSource = appendMessageContent(
        msg,
        'AI',
        textChunks,
        parts,
        foldBudget,
        preparation
      );
      if (aiSource != null) {
        foldedSourceMessages.push(aiSource);
      }

      let j = i + 1;
      while (j < messages.length && isToolMessage(messages[j])) {
        const toolSource = appendMessageContent(
          messages[j],
          'Tool',
          textChunks,
          parts,
          foldBudget,
          preparation
        );
        if (toolSource != null) {
          foldedSourceMessages.push(toolSource);
        }
        j++;
      }

      flushTextChunks(textChunks, parts);
      if (
        appendSyntheticProviderContextMessage(
          result,
          parts,
          foldedSourceMessages
        )
      ) {
        emitAgentLog(
          config,
          'warn',
          'format',
          'ensureThinkingBlockInMessages: injecting [Previous agent context] HumanMessage' +
            ` (${parts.length} msgs at index ${i}, no thinking block in chain)`
        );
      }
      i = j;
    } else {
      // Keep the message as is
      result.push(msg);
      i++;
    }
  }

  return result;
}

/** Whether a message carries tool content a tool-less agent cannot legally
 *  send. Covers every representation a provider converter will serialize back
 *  into a request: a ToolMessage, parsed `AIMessage.tool_calls`, raw
 *  `additional_kwargs.tool_calls` (OpenAI keeps calls here when the parsed
 *  array is empty), and `tool_use` / `tool_call` / `tool_result` content
 *  blocks (`@langchain/aws` and the Anthropic converter map these to Converse
 *  `toolUse` / `toolResult`). Missing the parent AI message is not just a
 *  passthrough: folding its ToolMessage alone would leave an orphan
 *  `assistant(tool_calls) -> user(...)` sequence. */
function messageHasToolContent(
  msg: BaseMessage,
  preserveNativeOpenAIResponses = false,
  preparation: ToolHistoryPreparation = createToolHistoryPreparation()
): boolean {
  if (isToolMessage(msg)) {
    return true;
  }
  const aiMsg = msg as AIMessage;
  if (aiMsg.tool_calls != null && aiMsg.tool_calls.length > 0) {
    return true;
  }
  const rawToolCalls = aiMsg.additional_kwargs.tool_calls;
  if (Array.isArray(rawToolCalls) && rawToolCalls.length > 0) {
    return true;
  }
  if (Array.isArray(msg.content)) {
    let hasOpenAINativeServerContent = false;
    for (const block of msg.content as ExtendedMessageContent[]) {
      const type = readFoldedDataProperty(block, 'type');
      const isNativeOpenAIServerContent =
        preserveNativeOpenAIResponses &&
        (type === 'server_tool_call' ||
          type === 'server_tool_call_result' ||
          type === 'server_tool_result');
      hasOpenAINativeServerContent ||= isNativeOpenAIServerContent;
      if (
        typeof block === 'object' &&
        !isNativeOpenAIServerContent &&
        (isProviderToolContentPart(block) ||
          readFoldedDataProperty(block, 'tool_call') != null)
      ) {
        return true;
      }
    }
    if (hasOpenAINativeServerContent) {
      return false;
    }
  }
  if (
    !preserveNativeOpenAIResponses &&
    getAuthoritativeResponsesToolOutput(msg, preparation) != null
  ) {
    return true;
  }
  return false;
}

/** Whether a message carries a tool RESULT: a ToolMessage, or a message whose
 *  content includes a `tool_result` block (the shape when a call/result pair is
 *  split as `AIMessage(tool_call)` + `HumanMessage(tool_result)`). Such a result
 *  belongs with the preceding tool call, so it is absorbed into the same fold
 *  and labelled as tool output. */
function isToolResultMessage(msg: BaseMessage): boolean {
  if (isToolMessage(msg)) {
    return true;
  }
  if (Array.isArray(msg.content)) {
    return (msg.content as ExtendedMessageContent[]).some(
      (block) =>
        typeof block === 'object' &&
        getProviderToolResultPartDescriptor(block) != null
    );
  }
  return false;
}

function getFoldedMessageRole(msg: BaseMessage): 'AI' | 'Tool' | 'User' {
  if (isToolMessage(msg)) {
    return 'Tool';
  }
  if (
    msg instanceof AIMessage ||
    msg instanceof AIMessageChunk ||
    ('role' in msg && msg.role === 'assistant')
  ) {
    return 'AI';
  }
  if (msg instanceof HumanMessage || ('role' in msg && msg.role === 'user')) {
    return 'User';
  }
  return isToolResultMessage(msg) ? 'Tool' : 'AI';
}

/**
 * Folds tool_use / tool_result content into plain text for an agent that binds
 * no tools.
 *
 * In a multi-agent graph, a tool-less destination still inherits the prior
 * agent's conversation history, which can contain toolUse/toolResult blocks.
 * Because it binds no tools, the model is invoked with no tool schema — and
 * Bedrock's Converse API rejects any request that carries toolUse/toolResult
 * blocks without a top-level toolConfig ("The toolConfig field must be defined
 * when using toolUse and toolResult content blocks"). Adding a dummy toolConfig
 * is not an option: AWS requires at least one tool, and it would expose a
 * capability the destination was intentionally denied.
 *
 * Each tool-call turn plus its trailing tool results (ToolMessages or
 * `tool_result` content blocks) is collapsed into a single `[Previous tool
 * interaction]` HumanMessage that preserves the tool name, arguments and result
 * as text (image blocks are kept as-is). Runs in a single pass: non-tool
 * messages pass through, `result` is allocated lazily on the first fold, and the
 * original array is returned unchanged when it holds no tool content (the common
 * fresh-tool-less-agent case).
 */
export function foldToolBlocksForToollessAgent(
  messages: BaseMessage[],
  config?: RunnableConfig,
  preserveNativeOpenAIResponses = false,
  preparation: ToolHistoryPreparation = createToolHistoryPreparation()
): BaseMessage[] {
  let result: BaseMessage[] | null = null;
  let foldedCount = 0;
  const foldBudget = createFoldContextBudget();
  let i = 0;
  while (i < messages.length) {
    const msg = messages[i];
    if (
      !messageHasToolContent(msg, preserveNativeOpenAIResponses, preparation)
    ) {
      result?.push(msg);
      i++;
      continue;
    }

    /** First fold — copy the untouched prefix once, then append from here. */
    if (result === null) {
      result = messages.slice(0, i);
    }

    const parts: MessageContentComplex[] = [];
    const textChunks: string[] = [];
    const foldedSourceMessages: FoldedSourceMessage[] = [];
    appendFoldedLine(textChunks, foldBudget, ['[Previous tool interaction]']);
    const initialSource = appendMessageContent(
      msg,
      getFoldedMessageRole(msg),
      textChunks,
      parts,
      foldBudget,
      preparation
    );
    if (initialSource != null) {
      foldedSourceMessages.push(initialSource);
    }
    foldedCount++;

    let j = i + 1;
    while (j < messages.length && isToolResultMessage(messages[j])) {
      const toolSource = appendMessageContent(
        messages[j],
        getFoldedMessageRole(messages[j]),
        textChunks,
        parts,
        foldBudget,
        preparation
      );
      if (toolSource != null) {
        foldedSourceMessages.push(toolSource);
      }
      foldedCount++;
      j++;
    }

    flushTextChunks(textChunks, parts);
    appendSyntheticProviderContextMessage(result, parts, foldedSourceMessages);
    i = j;
  }

  if (result === null) {
    return messages;
  }

  emitAgentLog(
    config,
    'warn',
    'format',
    `foldToolBlocksForToollessAgent: folded ${foldedCount} tool message(s) into text for a tool-less agent`
  );

  return result;
}

/**
 * Walks backwards from `currentIndex` through the message array to check
 * whether an earlier AI message in the same "chain" (no HumanMessage boundary)
 * contains a thinking/reasoning block.
 *
 * A "chain" is a contiguous sequence of AI + Tool messages with no intervening
 * HumanMessage. Bedrock reasoning models produce reasoning on the first AI
 * response, then issue follow-up tool calls with `content: ""` and no
 * reasoning block. These follow-ups are part of the same thinking-enabled
 * turn and should not be converted.
 */
function chainHasThinkingBlock(
  messages: BaseMessage[],
  currentIndex: number
): boolean {
  for (let k = currentIndex - 1; k >= 0; k--) {
    const prev = messages[k];

    // HumanMessage = turn boundary — stop searching
    if (
      prev instanceof HumanMessage ||
      ('role' in prev && (prev as any).role === 'user')
    ) {
      return false;
    }

    // Check AI messages for thinking/reasoning blocks
    const isPrevAI =
      prev instanceof AIMessage ||
      prev instanceof AIMessageChunk ||
      ('role' in prev && (prev as any).role === 'assistant');

    if (isPrevAI) {
      const prevAiMsg = prev as AIMessage | AIMessageChunk;

      if (Array.isArray(prevAiMsg.content) && prevAiMsg.content.length > 0) {
        const content = prevAiMsg.content as ExtendedMessageContent[];
        if (content.some((c) => isReasoningContentBlock(c))) {
          return true;
        }
      }

      // Bedrock also stores reasoning in additional_kwargs
      if (prevAiMsg.additional_kwargs.reasoning_content != null) {
        return true;
      }
    }

    // ToolMessages are part of the chain — keep walking back
  }

  return false;
}

// src/messages.ts
import { isProxy } from 'node:util/types';
import {
  AIMessage,
  BaseMessage,
  ToolMessage,
  HumanMessage,
  AIMessageChunk,
} from '@langchain/core/messages';
import type { ContentBlock as LangChainContentBlock } from '@langchain/core/messages';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type {
  ResponsesReplayPosition,
  ToolHistoryPreparation,
  OrderedToolHistoryProjection,
} from './toolHistoryProjection';
import type { ProviderMessageProvenancePart } from './provenance';
import type * as t from '@/types';
import {
  cloneToolMessageWithContent,
  compactToolContent,
  getToolContentCharLength,
  getBoundedCacheControlledTextToolContent,
  getBoundedSingleTextToolContent,
  getComputerCallOutputScreenshot,
  hasComputerCallOutputMarker,
  isComputerCallOutputMessage,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  OPENAI_RESPONSES_REPLAY_POSITIONS_KEY,
  createToolHistoryPreparation,
  getGeneratedImageMimeType,
  getResponsesHistorySource,
  isResponsesReplayPosition,
} from './toolHistoryProjection';
import {
  getProviderMessageProvenance,
  getProviderSourceMessageIds,
  hasBijectiveProviderContentPartMapping,
  setProviderMessageProvenance,
  stampSyntheticProviderMessage,
} from './provenance';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { isReasoningContentBlock } from './reasoningTypes';
import { stripAnthropicCacheControl } from './cache';
import { ContentTypes, Providers } from '@/common';
import { toLangChainContent } from './langchain';

type ReasoningSummary = { summary?: Array<{ text?: string }> };
type ReasoningDetail = { type?: string; text?: string };
type ReasoningAdditionalKwargs = {
  reasoning_content?: string | Partial<ReasoningSummary> | null;
  reasoning?: string | Partial<ReasoningSummary> | null;
  reasoning_details?: ReasoningDetail[] | null;
};

export function getConverseOverrideMessage({
  userMessage,
  lastMessageX,
  lastMessageY,
}: {
  userMessage: string[];
  lastMessageX: AIMessageChunk | null;
  lastMessageY: ToolMessage;
}): HumanMessage {
  const content = `
User: ${userMessage[1]}

---
# YOU HAVE ALREADY RESPONDED TO THE LATEST USER MESSAGE:

# Observations:
- ${lastMessageX?.content}

# Tool Calls:
- ${lastMessageX?.tool_calls?.join('\n- ')}

# Tool Responses:
- ${lastMessageY.content}
`;

  return new HumanMessage(content);
}

const _allowedTypes = ['image_url', 'text', 'tool_use', 'tool_result'];
const allowedTypesByProvider: Record<string, string[]> = {
  default: _allowedTypes,
  [Providers.ANTHROPIC]: [
    ..._allowedTypes,
    'thinking',
    'redacted_thinking',
    'server_tool_use',
    'web_search_tool_result',
    'web_search_result',
  ],
  [Providers.BEDROCK]: [..._allowedTypes, 'reasoning_content'],
  [Providers.OPENAI]: _allowedTypes,
};

const modifyContent = ({
  provider,
  messageType,
  content,
}: {
  provider: t.ProviderName;
  messageType: string;
  content: t.ExtendedMessageContent[];
}): (t.ExtendedMessageContent | null)[] => {
  const allowedTypes =
    allowedTypesByProvider[provider] ?? allowedTypesByProvider.default;
  return content.map((item: t.ExtendedMessageContent | null) => {
    if (
      item &&
      typeof item === 'object' &&
      'type' in item &&
      item.type != null &&
      item.type
    ) {
      let newType = item.type;
      if (newType.endsWith('_delta')) {
        newType = newType.replace('_delta', '');
      }
      if (!allowedTypes.includes(newType)) {
        newType = 'text';
      }

      /* Handle the edge case for empty object 'tool_use' input in AI messages */
      if (
        messageType === 'ai' &&
        newType === 'tool_use' &&
        'input' in item &&
        item.input === ''
      ) {
        return { ...item, type: newType, input: '{}' };
      }

      return { ...item, type: newType };
    }
    return item;
  });
};

type ContentBlock =
  | Partial<t.BedrockReasoningContentText>
  | t.MessageDeltaUpdate;

function reduceBlocks(blocks: ContentBlock[]): ContentBlock[] {
  const reduced: ContentBlock[] = [];

  for (const block of blocks) {
    const lastBlock = reduced[reduced.length - 1] as ContentBlock | undefined;

    // Merge consecutive 'reasoning_content'
    if (
      block.type === 'reasoning_content' &&
      lastBlock?.type === 'reasoning_content'
    ) {
      // append text if exists
      if (block.reasoningText?.text != null && block.reasoningText.text) {
        (
          lastBlock.reasoningText as t.BedrockReasoningContentText['reasoningText']
        ).text =
          (lastBlock.reasoningText?.text ?? '') + block.reasoningText.text;
      }
      // preserve the signature if exists
      if (
        block.reasoningText?.signature != null &&
        block.reasoningText.signature
      ) {
        (
          lastBlock.reasoningText as t.BedrockReasoningContentText['reasoningText']
        ).signature = block.reasoningText.signature;
      }
    }
    // Merge consecutive 'text'
    else if (block.type === 'text' && lastBlock?.type === 'text') {
      lastBlock.text += block.text;
    }
    // add a new block as it's a different type or first element
    else {
      // deep copy to avoid mutation of original
      reduced.push(JSON.parse(JSON.stringify(block)));
    }
  }

  return reduced;
}

function getReasoningText(
  value: string | Partial<ReasoningSummary> | null | undefined
): string | undefined {
  if (typeof value === 'string') {
    return value !== '' ? value : undefined;
  }
  const summaryText = value?.summary
    ?.map((summary) => summary.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return summaryText != null && summaryText !== '' ? summaryText : undefined;
}

function getReasoningDetailsText(
  value: ReasoningDetail[] | null | undefined
): string | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const reasoningText = value
    .filter((detail) => detail.type === 'reasoning.text')
    .map((detail) => detail.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return reasoningText !== '' ? reasoningText : undefined;
}

function getAdditionalReasoningContent(
  message: BaseMessage
): string | undefined {
  const additionalKwargs = message.additional_kwargs as
    | ReasoningAdditionalKwargs
    | undefined;
  if (additionalKwargs == null) {
    return undefined;
  }

  const reasoningContent = getReasoningText(additionalKwargs.reasoning_content);
  if (reasoningContent != null) {
    return reasoningContent;
  }

  const reasoning = getReasoningText(additionalKwargs.reasoning);
  if (reasoning != null) {
    return reasoning;
  }

  return getReasoningDetailsText(additionalKwargs.reasoning_details);
}

/**
 * True when a content block is a provider reasoning block: Anthropic
 * `thinking` / `redacted_thinking`, Google `reasoning`, Bedrock
 * `reasoning_content`, LibreChat `think`, and their known streamed variants.
 *
 * Shared rather than re-derived because two very different readers must agree
 * on the same answer: the stream classifier, which decides what the host has
 * been shown, and the preempt gate, which decides whether an accumulated turn
 * holds anything worth keeping. A turn the host renders as thinking-only must
 * never look like a visible answer to the gate.
 *
 * Exact matching prevents unrelated custom block types that happen to share a
 * prefix from being silently classified as hidden reasoning.
 */
export { isReasoningContentBlock } from './reasoningTypes';

function hasReasoningContent(content: BaseMessage['content']): boolean {
  if (!Array.isArray(content)) {
    return false;
  }
  return content.some(
    (item) => typeof item === 'object' && isReasoningContentBlock(item)
  );
}

export function modifyDeltaProperties(
  provider: t.ProviderName,
  obj?: AIMessageChunk
): AIMessageChunk | undefined {
  if (!obj || typeof obj !== 'object') return obj;

  const messageType = (obj as Partial<AIMessageChunk>)._getType
    ? obj._getType()
    : '';

  if (provider === Providers.BEDROCK && Array.isArray(obj.content)) {
    obj.content = toLangChainContent(
      reduceBlocks(obj.content as ContentBlock[])
    );
  }
  if (Array.isArray(obj.content)) {
    obj.content = toLangChainContent(
      modifyContent({
        provider,
        messageType,
        content: obj.content as t.ExtendedMessageContent[],
      }) as t.MessageContentComplex[]
    );
  }
  if (
    (obj as Partial<AIMessageChunk>).lc_kwargs &&
    Array.isArray(obj.lc_kwargs.content)
  ) {
    if (provider === Providers.BEDROCK) {
      obj.lc_kwargs.content = reduceBlocks(
        obj.lc_kwargs.content as ContentBlock[]
      );
    }
    obj.lc_kwargs.content = modifyContent({
      provider,
      messageType,
      content: obj.lc_kwargs.content,
    });
  }
  return obj;
}

export function formatAnthropicMessage(message: AIMessageChunk): AIMessage {
  if (!message.tool_calls || message.tool_calls.length === 0) {
    return new AIMessage({ content: toLangChainContent(message.content) });
  }

  const toolCallMap = new Map(message.tool_calls.map((tc) => [tc.id, tc]));
  let formattedContent: string | t.ExtendedMessageContent[];

  if (Array.isArray(message.content)) {
    formattedContent = message.content.reduce<t.ExtendedMessageContent[]>(
      (acc, item) => {
        if (typeof item === 'object') {
          const extendedItem = item as t.ExtendedMessageContent;
          if (
            extendedItem.type === 'text' &&
            extendedItem.text != null &&
            extendedItem.text
          ) {
            acc.push({ type: 'text', text: extendedItem.text });
          } else if (
            extendedItem.type === 'tool_use' &&
            extendedItem.id != null &&
            extendedItem.id
          ) {
            const toolCall = toolCallMap.get(extendedItem.id);
            if (toolCall) {
              acc.push({
                type: 'tool_use',
                id: extendedItem.id,
                name: toolCall.name,
                input: toolCall.args as unknown as string,
              });
            }
          } else if (
            'input' in extendedItem &&
            extendedItem.input != null &&
            extendedItem.input
          ) {
            try {
              const parsedInput = JSON.parse(extendedItem.input);
              const toolCall = message.tool_calls?.find(
                (tc) => tc.args.input === parsedInput.input
              );
              if (toolCall) {
                acc.push({
                  type: 'tool_use',
                  id: toolCall.id,
                  name: toolCall.name,
                  input: toolCall.args as unknown as string,
                });
              }
            } catch {
              if (extendedItem.input) {
                acc.push({ type: 'text', text: extendedItem.input });
              }
            }
          }
        } else if (typeof item === 'string') {
          acc.push({ type: 'text', text: item });
        }
        return acc;
      },
      []
    );
  } else if (typeof message.content === 'string') {
    formattedContent = message.content;
  } else {
    formattedContent = [];
  }

  // const formattedToolCalls: ToolCall[] = message.tool_calls.map(toolCall => ({
  //   id: toolCall.id ?? '',
  //   name: toolCall.name,
  //   args: toolCall.args,
  //   type: 'tool_call',
  // }));

  const formattedToolCalls: t.AgentToolCall[] = message.tool_calls.map(
    (toolCall) => ({
      id: toolCall.id ?? '',
      type: 'function',
      function: {
        name: toolCall.name,
        arguments: toolCall.args,
      },
    })
  );

  return new AIMessage({
    content: toLangChainContent(formattedContent),
    tool_calls: formattedToolCalls as ToolCall[],
    additional_kwargs: {
      ...message.additional_kwargs,
    },
  });
}

export function convertMessagesToContent(
  messages: BaseMessage[]
): t.MessageContentComplex[] {
  const processedContent: t.MessageContentComplex[] = [];

  const addToolCallBoundary = (): number => {
    processedContent.push({ type: ContentTypes.TEXT, text: '' });
    return processedContent.length - 1;
  };

  const addContentPart = (message: BaseMessage | null): number | undefined => {
    const content =
      message?.lc_kwargs.content != null
        ? message.lc_kwargs.content
        : message?.content;
    if (content === undefined) {
      return undefined;
    }
    const reasoningContent =
      message?._getType() === 'ai' && !hasReasoningContent(content)
        ? getAdditionalReasoningContent(message)
        : undefined;
    if (reasoningContent != null) {
      processedContent.push({
        type: ContentTypes.THINK,
        think: reasoningContent,
      });
    }
    if (typeof content === 'string') {
      if (content === '') {
        return undefined;
      }
      processedContent.push({
        type: ContentTypes.TEXT,
        text: content,
      });
      return processedContent.length - 1;
    } else if (Array.isArray(content)) {
      let textContentIndex: number | undefined;
      for (const item of content) {
        if (item == null || item.type === 'tool_use') {
          continue;
        }
        processedContent.push(item);
        if (item.type === ContentTypes.TEXT) {
          textContentIndex = processedContent.length - 1;
        }
      }
      return textContentIndex;
    }
    return undefined;
  };

  let currentAIMessageIndex = -1;
  const toolCallMap = new Map<string, t.CustomToolCall>();

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i] as BaseMessage | null;
    const messageType = message?._getType();

    if (
      messageType === 'ai' &&
      ((message as AIMessage).tool_calls?.length ?? 0) > 0
    ) {
      const tool_calls = (message as AIMessage).tool_calls || [];
      for (const tool_call of tool_calls) {
        if (tool_call.id == null || !tool_call.id) {
          continue;
        }

        toolCallMap.set(tool_call.id, tool_call);
      }

      currentAIMessageIndex = addContentPart(message) ?? addToolCallBoundary();
      continue;
    } else if (
      messageType === 'tool' &&
      (message as ToolMessage).tool_call_id
    ) {
      const id = (message as ToolMessage).tool_call_id;
      const output = (message as ToolMessage).content;
      const tool_call = toolCallMap.get(id);
      if (currentAIMessageIndex === -1) {
        processedContent.push({ type: 'text', text: '' });
        currentAIMessageIndex = processedContent.length - 1;
      }
      const contentPart = processedContent[currentAIMessageIndex];
      processedContent.push({
        type: 'tool_call',
        tool_call: Object.assign({}, tool_call, { output }),
      });
      const tool_call_ids = contentPart.tool_call_ids || [];
      tool_call_ids.push(id);
      contentPart.tool_call_ids = tool_call_ids;
      continue;
    } else if (messageType !== 'ai') {
      continue;
    }

    addContentPart(message);
  }

  return processedContent;
}

function stringifyToolMessageContent(
  content: ToolMessage['content'] | null | undefined
): string {
  return content == null ? '' : String(content);
}

function appendContentBlocks(
  target: t.MessageContentComplex[],
  content: BaseMessage['content']
): void {
  if (typeof content === 'string') {
    target.push({ type: ContentTypes.TEXT, text: content });
    return;
  }
  for (const block of content) {
    target.push(block as t.MessageContentComplex);
  }
}

interface BoundedContentAccumulator {
  readonly blocks: t.MessageContentComplex[];
  hasContent: boolean;
  remainingChars: number;
}

function createBoundedContentAccumulator(
  maxChars: number
): BoundedContentAccumulator {
  return {
    blocks: [],
    hasContent: false,
    remainingChars: Number.isFinite(maxChars)
      ? Math.max(0, Math.floor(maxChars))
      : Number.MAX_SAFE_INTEGER,
  };
}

/** Appends a segment only when at least one of its provider-visible bytes fits.
 * Each segment is compacted against the remaining budget before combination,
 * so a later segment cannot be credited merely because recompression replaced
 * earlier bytes with a truncation notice. */
function appendBoundedContentSegment(
  accumulator: BoundedContentAccumulator,
  next: unknown
): { contributed: boolean; complete: boolean } {
  const separatorChars = accumulator.hasContent ? 1 : 0;
  const availableChars = accumulator.remainingChars - separatorChars;
  const originalChars = getToolContentCharLength(next);
  if (originalChars <= 0) {
    return { contributed: false, complete: true };
  }
  if (availableChars <= 0) {
    return { contributed: false, complete: false };
  }
  const compactedNext = compactToolContent(next, availableChars);
  const boundedNext = compactedNext.content;
  const appendedChars = Math.min(
    availableChars,
    getToolContentCharLength(boundedNext)
  );
  if (appendedChars <= 0) {
    return { contributed: false, complete: false };
  }
  appendContentBlocks(accumulator.blocks, boundedNext);
  accumulator.hasContent = true;
  accumulator.remainingChars -= separatorChars + appendedChars;
  return {
    contributed: true,
    complete: !compactedNext.changed && originalChars <= availableChars,
  };
}

function appendBoundedContentContribution(
  accumulator: BoundedContentAccumulator,
  next: unknown
): {
  contributed: boolean;
  complete: boolean;
  retainedContentPartIndices?: ReadonlySet<number>;
} {
  if (!Array.isArray(next)) {
    return appendBoundedContentSegment(accumulator, next);
  }
  const retainedContentPartIndices = new Set<number>();
  let complete = true;
  for (let index = 0; index < next.length; index++) {
    const projection = appendBoundedContentSegment(accumulator, [next[index]]);
    if (!projection.complete) {
      complete = false;
    }
    if (!projection.contributed) {
      const separatorChars = accumulator.hasContent ? 1 : 0;
      if (accumulator.remainingChars <= separatorChars) {
        if (index < next.length - 1) {
          complete = false;
        }
        break;
      }
      continue;
    }
    retainedContentPartIndices.add(index);
  }
  return {
    contributed: retainedContentPartIndices.size > 0,
    complete,
    retainedContentPartIndices,
  };
}

function retainProjectedContentPartProvenance(
  message: BaseMessage,
  parts: readonly ProviderMessageProvenancePart[],
  retainedContentPartIndices: ReadonlySet<number> | undefined,
  complete: boolean
): ProviderMessageProvenancePart[] {
  if (
    complete &&
    (retainedContentPartIndices == null ||
      !Array.isArray(message.content) ||
      retainedContentPartIndices.size === message.content.length)
  ) {
    return [...parts];
  }
  const mapsOneToOne = hasBijectiveProviderContentPartMapping(
    parts,
    message.content.length
  );
  const distinctSourceMessageIds = new Set<string>();
  for (const part of parts) {
    if (part.sourceMessageId != null) {
      distinctSourceMessageIds.add(part.sourceMessageId);
    }
  }
  if (!Array.isArray(message.content)) {
    const soleSourceMessageId =
      distinctSourceMessageIds.size === 1
        ? distinctSourceMessageIds.values().next().value
        : undefined;
    return [
      {
        attribution: parts[0]?.attribution ?? 'tool',
        ...(soleSourceMessageId != null && {
          sourceMessageId: soleSourceMessageId,
        }),
      },
    ];
  }
  const retained: ProviderMessageProvenancePart[] = [];
  const appendUnindexedAttribution = (
    part: ProviderMessageProvenancePart
  ): void => {
    const sourceMessageId =
      distinctSourceMessageIds.size <= 1 ? part.sourceMessageId : undefined;
    if (retained.length > 0) {
      const previous = retained[retained.length - 1];
      if (
        previous.attribution === part.attribution &&
        previous.sourceMessageId === sourceMessageId &&
        previous.sourceContentPartIndices == null
      ) {
        return;
      }
    }
    retained.push({
      attribution: part.attribution,
      ...(sourceMessageId != null && { sourceMessageId }),
    });
  };
  for (const part of parts) {
    if (part.sourceContentPartIndices == null) {
      appendUnindexedAttribution(part);
      continue;
    }
    if (!mapsOneToOne || retainedContentPartIndices == null) {
      appendUnindexedAttribution(part);
      continue;
    }
    const sourceContentPartIndices: number[] = [];
    for (const sourceContentPartIndex of part.sourceContentPartIndices) {
      if (retainedContentPartIndices.has(sourceContentPartIndex)) {
        sourceContentPartIndices.push(sourceContentPartIndex);
      }
    }
    if (sourceContentPartIndices.length > 0) {
      retained.push({ ...part, sourceContentPartIndices });
    }
  }
  return retained.length > 0
    ? retained
    : [{ attribution: parts[0]?.attribution ?? 'tool' }];
}

function cloneAIMessageWithToolCalls(
  message: AIMessage,
  toolCalls: ToolCall[],
  removedCallIds: ReadonlySet<string>
): AIMessage {
  const descriptors = Object.getOwnPropertyDescriptors(message) as Record<
    string,
    PropertyDescriptor | undefined
  >;
  let descriptor = descriptors.tool_calls;
  descriptors.tool_calls = {
    configurable: descriptor?.configurable ?? true,
    enumerable: descriptor?.enumerable ?? true,
    value: toolCalls,
    writable: descriptor?.writable ?? true,
  };
  const toolCallChunks = descriptors.tool_call_chunks?.value as
    | ToolCallChunk[]
    | undefined;
  if (Array.isArray(toolCallChunks)) {
    descriptor = descriptors.tool_call_chunks;
    descriptors.tool_call_chunks = {
      configurable: descriptor?.configurable ?? true,
      enumerable: descriptor?.enumerable ?? true,
      value: toolCallChunks.filter(
        (chunk) => typeof chunk.id !== 'string' || !removedCallIds.has(chunk.id)
      ),
      writable: descriptor?.writable ?? true,
    };
  }
  return Object.create(
    Object.getPrototypeOf(message),
    descriptors as PropertyDescriptorMap
  ) as AIMessage;
}

function cloneAIMessageWithContent(
  message: AIMessage,
  content: AIMessage['content']
): AIMessage {
  const descriptors = Object.getOwnPropertyDescriptors(message) as Record<
    string,
    PropertyDescriptor | undefined
  >;
  const descriptor = descriptors.content;
  descriptors.content = {
    configurable: descriptor?.configurable ?? true,
    enumerable: descriptor?.enumerable ?? true,
    value: content,
    writable: descriptor?.writable ?? true,
  };
  const lcKwargs = descriptors.lc_kwargs;
  if (
    lcKwargs != null &&
    'value' in lcKwargs &&
    typeof lcKwargs.value === 'object' &&
    lcKwargs.value != null
  ) {
    descriptors.lc_kwargs = {
      ...lcKwargs,
      value: {
        ...(lcKwargs.value as Record<string, unknown>),
        ...(message.response_metadata.output_version === 'v1'
          ? { content: undefined, contentBlocks: content }
          : { content }),
      },
    };
  }
  return Object.create(
    Object.getPrototypeOf(message),
    descriptors as PropertyDescriptorMap
  ) as AIMessage;
}

function cloneAIMessageWithResponsesReplayState(
  message: AIMessage,
  content: AIMessage['content'],
  id: string | undefined,
  additionalKwargs: AIMessage['additional_kwargs'],
  responseMetadata: AIMessage['response_metadata']
): AIMessage {
  const descriptors = Object.getOwnPropertyDescriptors(message) as Record<
    string,
    PropertyDescriptor | undefined
  >;
  const replacements = {
    content,
    id,
    additional_kwargs: additionalKwargs,
    response_metadata: responseMetadata,
  };
  for (const [key, value] of Object.entries(replacements)) {
    const descriptor = descriptors[key];
    descriptors[key] = {
      configurable: descriptor?.configurable ?? true,
      enumerable: descriptor?.enumerable ?? true,
      value,
      writable: descriptor?.writable ?? true,
    };
  }
  const lcKwargs = descriptors.lc_kwargs;
  if (
    lcKwargs != null &&
    'value' in lcKwargs &&
    typeof lcKwargs.value === 'object' &&
    lcKwargs.value != null
  ) {
    descriptors.lc_kwargs = {
      ...lcKwargs,
      value: {
        ...(lcKwargs.value as Record<string, unknown>),
        ...replacements,
        ...(responseMetadata.output_version === 'v1'
          ? { content: undefined, contentBlocks: content }
          : {}),
      },
    };
  }
  return Object.create(
    Object.getPrototypeOf(message),
    descriptors as PropertyDescriptorMap
  ) as AIMessage;
}

function hasReplayableEncryptedReasoning(reasoning: unknown): boolean {
  if (reasoning == null || typeof reasoning !== 'object') {
    return false;
  }
  const item = reasoning as {
    encrypted_content?: unknown;
    id?: unknown;
    status?: unknown;
    summary?: unknown;
    type?: unknown;
  };
  return (
    item.type === 'reasoning' &&
    typeof item.id === 'string' &&
    item.id.length > 0 &&
    Array.isArray(item.summary) &&
    typeof item.encrypted_content === 'string' &&
    item.encrypted_content.length > 0 &&
    (item.status === undefined ||
      item.status === 'completed' ||
      item.status === 'incomplete')
  );
}

type ResponsesReplayProjection = 'fallback' | 'native';

export { OPENAI_RESPONSES_REPLAY_POSITIONS_KEY } from './toolHistoryProjection';

/** Reasoning item currently streaming, so a terminal ciphertext seals against its own id. */
export const OPENAI_RESPONSES_ACTIVE_REASONING_ID_KEY =
  '__openai_responses_active_reasoning_id__';

export type { ResponsesReplayPosition } from './toolHistoryProjection';

type CompletedGeneratedImages = {
  blocks: PositionedResponsesReplayBlock[];
  data: Set<string>;
  ids: Set<string>;
};

type ResponsesReplayArtifacts = {
  generatedImages: CompletedGeneratedImages;
  messagePositions: ResponsesReplayPosition[];
  positionsByItemId: Map<string, ResponsesReplayItemPosition>;
  positionedServerToolResultIds: Set<string>;
  serverToolResults: PositionedResponsesReplayBlock[];
  textPositions: ResponsesReplayPosition[];
};

type ResponsesReplayItemPosition = Pick<
  PositionedResponsesReplayBlock,
  'outputIndex' | 'textIndex'
>;

type PositionedResponsesReplayBlock = {
  block: LangChainContentBlock.Standard;
  outputIndex: number;
  subIndex: number;
  textIndex?: number;
};

type PositionedResponsesImage = {
  block: LangChainContentBlock.Multimodal.Image;
  textIndex: number;
};

type PositionedResponsesImages = {
  blocks: PositionedResponsesImage[];
  textCount: number;
};

function isProviderGeneratedImageBlock(
  block: LangChainContentBlock.Multimodal.Image,
  generatedImages: CompletedGeneratedImages,
  allowDataFallback = true
): boolean {
  if (typeof block.id === 'string' && block.id.length > 0) {
    return generatedImages.ids.has(block.id);
  }
  return (
    allowDataFallback &&
    typeof block.data === 'string' &&
    generatedImages.data.has(block.data) &&
    block.metadata != null &&
    typeof block.metadata === 'object' &&
    block.metadata.status === 'completed'
  );
}

function hasMeaningfulServerToolOutput(output: unknown): boolean {
  if (output == null || output === '') {
    return false;
  }
  if (Array.isArray(output)) {
    return output.length > 0;
  }
  if (typeof output !== 'object') {
    return true;
  }
  try {
    return Object.keys(output).length > 0;
  } catch {
    return false;
  }
}

function createServerToolResultExtras(toolName?: string): {
  librechatServerToolResult: { toolName?: string };
} {
  return {
    librechatServerToolResult: {
      ...(toolName != null ? { toolName } : {}),
    },
  };
}

function createNeutralServerToolResult(
  output: unknown,
  status: 'error' | 'success',
  maxChars: number,
  toolName?: string
): LangChainContentBlock.Text | undefined {
  if (!hasMeaningfulServerToolOutput(output)) {
    return undefined;
  }
  return {
    type: 'text',
    text: serializeToolContentBounded(
      {
        serverToolResult: {
          librechatResponsesReplay: true,
          ...(toolName != null ? { toolName } : {}),
          status,
          output,
        },
      },
      maxChars
    ),
    extras: createServerToolResultExtras(toolName),
  };
}

function isCompleteToolStreamContentBlock(block: unknown): boolean {
  if (block == null || typeof block !== 'object') {
    return true;
  }
  try {
    if (isProxy(block)) {
      return false;
    }
    const type = Object.getOwnPropertyDescriptor(block, 'type');
    if (type == null) {
      return true;
    }
    if (type.enumerable !== true || !('value' in type)) {
      return false;
    }
    if (type.value !== 'text') {
      return true;
    }
    const text = Object.getOwnPropertyDescriptor(block, 'text');
    return (
      text?.enumerable === true &&
      'value' in text &&
      typeof text.value === 'string' &&
      text.value !== ''
    );
  } catch {
    return false;
  }
}

/** Drops incomplete streamed text blocks before provider-facing tool shaping. */
export function dropIncompleteToolStreamContent(
  content: AIMessage['content']
): AIMessage['content'] {
  if (!Array.isArray(content)) {
    return content;
  }
  const completeContent = content.filter(isCompleteToolStreamContentBlock);
  return completeContent.length === content.length
    ? content
    : toLangChainContent(completeContent);
}

function projectPreemptedResponsesV1Content(
  content: AIMessage['content'],
  reasoning: unknown,
  maxChars: number,
  replayProjection: ResponsesReplayProjection,
  generatedImages?: CompletedGeneratedImages,
  originalImages?: PositionedResponsesImages,
  serverToolResults?: PositionedResponsesReplayBlock[],
  positionedServerToolResultIds?: ReadonlySet<string>,
  reasoningPosition?: ResponsesReplayItemPosition,
  textPositions?: readonly ResponsesReplayPosition[],
  messagePositions?: readonly ResponsesReplayPosition[]
): {
  content: AIMessage['content'];
  preservedServerToolResult: boolean;
} {
  if (!Array.isArray(content)) {
    return { content, preservedServerToolResult: false };
  }

  const contentBlocks = content as LangChainContentBlock.Standard[];
  const encryptedReasoning =
    replayProjection === 'native' && hasReplayableEncryptedReasoning(reasoning)
      ? reasoning
      : undefined;
  const positionedReasoning: PositionedResponsesReplayBlock | undefined =
    encryptedReasoning != null && reasoningPosition != null
      ? {
        block: {
          type: 'non_standard' as const,
          value: encryptedReasoning,
        },
        ...reasoningPosition,
        subIndex: 0,
      }
      : undefined;
  const projected: LangChainContentBlock.Standard[] = [];
  let changed = false;
  let hasEncryptedReasoning = false;
  let preservedServerToolResult = false;
  let reasoningInsertionIndex: number | undefined;
  let generatedImageIndex = 0;
  let originalImageIndex = 0;
  let reasoningPending = positionedReasoning;
  let serverToolResultIndex = 0;
  let sourceTextIndex = 0;
  const serverToolNamesByCallId = new Map<string, string>();

  const appendOriginalImages = (throughTextIndex?: number): void => {
    if (replayProjection !== 'native' || originalImages == null) {
      return;
    }
    while (originalImageIndex < originalImages.blocks.length) {
      const positionedImage = originalImages.blocks[originalImageIndex];
      if (
        throughTextIndex != null &&
        positionedImage.textIndex > throughTextIndex
      ) {
        return;
      }
      originalImageIndex++;
      if (
        generatedImages != null &&
        isProviderGeneratedImageBlock(
          positionedImage.block,
          generatedImages,
          false
        )
      ) {
        continue;
      }
      projected.push(positionedImage.block);
      changed = true;
    }
  };

  const appendReplayBlocks = (
    throughTextIndex?: number,
    beforeOutputIndex?: number
  ): void => {
    for (;;) {
      const generatedImage = generatedImages?.blocks[generatedImageIndex];
      const serverToolResult = serverToolResults?.[serverToolResultIndex];
      const candidates: Array<{
        kind: 'generatedImage' | 'reasoning' | 'serverToolResult';
        value: PositionedResponsesReplayBlock;
      }> = [];
      if (generatedImage != null) {
        candidates.push({ kind: 'generatedImage', value: generatedImage });
      }
      if (serverToolResult != null) {
        candidates.push({ kind: 'serverToolResult', value: serverToolResult });
      }
      if (reasoningPending != null) {
        candidates.push({ kind: 'reasoning', value: reasoningPending });
      }
      if (candidates.length === 0) {
        return;
      }
      candidates.sort((a, b) =>
        comparePositionedResponsesReplayBlocks(a.value, b.value)
      );
      const selected = candidates[0];
      const positionedResult = selected.value;
      if (
        throughTextIndex != null &&
        (positionedResult.textIndex == null ||
          positionedResult.textIndex > throughTextIndex)
      ) {
        return;
      }
      if (
        beforeOutputIndex != null &&
        positionedResult.outputIndex >= beforeOutputIndex
      ) {
        return;
      }
      if (selected.kind === 'generatedImage') {
        generatedImageIndex++;
      } else if (selected.kind === 'serverToolResult') {
        serverToolResultIndex++;
      } else {
        reasoningPending = undefined;
      }
      if (
        replayProjection === 'fallback' &&
        positionedResult.block.type === 'text'
      ) {
        projected.push({
          type: 'text',
          text: positionedResult.block.text,
        });
      } else {
        projected.push(positionedResult.block);
      }
      changed = true;
      preservedServerToolResult ||= selected.kind === 'serverToolResult';
    }
  };

  const appendPositionedOriginalImages = (throughTextIndex?: number): void => {
    const nextOriginalImage = originalImages?.blocks[originalImageIndex];
    if (
      nextOriginalImage == null ||
      (throughTextIndex != null &&
        nextOriginalImage.textIndex > throughTextIndex)
    ) {
      return;
    }
    const followingTextPosition = textPositions?.[nextOriginalImage.textIndex];
    const imageOnlyMessagePosition =
      originalImages?.textCount === 0 ? messagePositions?.[0] : undefined;
    const outputBoundary = followingTextPosition ?? imageOnlyMessagePosition;
    if (outputBoundary != null) {
      appendReplayBlocks(
        nextOriginalImage.textIndex,
        outputBoundary.outputIndex
      );
    }
    appendOriginalImages(throughTextIndex);
  };

  for (let i = 0; i < contentBlocks.length; i++) {
    const block = contentBlocks[i];
    if (!isCompleteToolStreamContentBlock(block)) {
      changed = true;
      continue;
    }
    if (block.type === 'reasoning') {
      reasoningInsertionIndex ??= projected.length;
      changed = true;
      continue;
    }
    if (block.type === 'non_standard') {
      if (
        replayProjection === 'native' &&
        !hasEncryptedReasoning &&
        hasReplayableEncryptedReasoning(block.value)
      ) {
        hasEncryptedReasoning = true;
        if (positionedReasoning == null) {
          projected.push(block);
        } else {
          changed = true;
        }
      } else {
        changed = true;
      }
      continue;
    }
    if (block.type === 'text') {
      appendPositionedOriginalImages(sourceTextIndex);
      appendReplayBlocks(sourceTextIndex);
      if (replayProjection === 'fallback') {
        projected.push({ type: 'text', text: block.text });
        changed = true;
      } else {
        projected.push(block);
      }
      sourceTextIndex++;
      continue;
    }
    if (originalImages != null && sourceTextIndex >= originalImages.textCount) {
      appendPositionedOriginalImages();
    }
    if (
      block.type === 'server_tool_call' ||
      block.type === 'server_tool_call_chunk'
    ) {
      if (
        typeof block.id === 'string' &&
        block.id.length > 0 &&
        typeof block.name === 'string' &&
        block.name.length > 0
      ) {
        serverToolNamesByCallId.set(block.id, block.name);
      }
      changed = true;
      continue;
    }
    if (block.type === 'server_tool_call_result') {
      if (positionedServerToolResultIds?.has(block.toolCallId) === true) {
        changed = true;
        continue;
      }
      const result = createNeutralServerToolResult(
        block.output,
        block.status,
        maxChars,
        serverToolNamesByCallId.get(block.toolCallId)
      );
      changed = true;
      if (result != null) {
        projected.push(
          replayProjection === 'fallback'
            ? { type: 'text', text: result.text }
            : result
        );
        preservedServerToolResult = true;
      }
      continue;
    }
    if (block.type === 'image') {
      if (
        replayProjection === 'fallback' ||
        (generatedImages != null &&
          isProviderGeneratedImageBlock(block, generatedImages))
      ) {
        changed = true;
        continue;
      }
    }
    projected.push(block);
  }

  if (
    encryptedReasoning != null &&
    !hasEncryptedReasoning &&
    positionedReasoning == null
  ) {
    const reasoningBlock: LangChainContentBlock.Standard = {
      type: 'non_standard',
      value: encryptedReasoning,
    };
    projected.splice(reasoningInsertionIndex ?? 0, 0, reasoningBlock);
    changed = true;
  }
  appendPositionedOriginalImages();
  appendReplayBlocks();
  return {
    content: changed ? toLangChainContent(projected) : content,
    preservedServerToolResult,
  };
}

function getAuthoritativeResponsesOutput(message: AIMessage): unknown[] {
  return getResponsesHistorySource(message)?.items ?? [];
}

function getResponsesReplayItemKey(
  item: Record<string, unknown>
): string | undefined {
  if (typeof item.id === 'string' && item.id.length > 0) {
    return item.id;
  }
  return typeof item.call_id === 'string' && item.call_id.length > 0
    ? item.call_id
    : undefined;
}

function comparePositionedResponsesReplayBlocks(
  a: PositionedResponsesReplayBlock,
  b: PositionedResponsesReplayBlock
): number {
  return (
    (a.textIndex ?? Number.MAX_SAFE_INTEGER) -
      (b.textIndex ?? Number.MAX_SAFE_INTEGER) ||
    a.outputIndex - b.outputIndex ||
    a.subIndex - b.subIndex
  );
}

const RESPONSES_REPLAY_OUTPUT_TOOL_NAMES = {
  apply_patch_call_output: 'apply_patch',
  local_shell_call_output: 'local_shell',
  shell_call_output: 'shell',
} as const;

function getResponsesReplayArtifacts(
  output: readonly unknown[],
  maxChars: number,
  replayProjection: ResponsesReplayProjection,
  replayPositionValue?: unknown,
  history?: OrderedToolHistoryProjection
): ResponsesReplayArtifacts {
  const blocks: PositionedResponsesReplayBlock[] = [];
  const data = new Set<string>();
  const ids = new Set<string>();
  const emittedData = new Set<string>();
  const emittedIds = new Set<string>();
  const positionedServerToolResultIds = new Set<string>();
  const rawOutputIndices = new Map<Record<string, unknown>, number>();
  const serverToolResults: PositionedResponsesReplayBlock[] = [];
  const replayPositions = Array.isArray(replayPositionValue)
    ? replayPositionValue.filter(isResponsesReplayPosition)
    : [];
  const authoritativeOutputIndicesByItemId = new Map<string, number>();
  const messagePositionsByKey = new Map<string, ResponsesReplayPosition>();
  const outputPositionsByItemId = new Map<string, ResponsesReplayPosition>();
  const textPositionsByKey = new Map<string, ResponsesReplayPosition>();
  for (const position of replayPositions) {
    if (position.kind === 'output' || position.kind === 'reasoning') {
      outputPositionsByItemId.set(position.itemId, position);
      continue;
    }
    if (position.kind === 'message') {
      messagePositionsByKey.set(
        `${position.itemId}:${position.outputIndex}`,
        position
      );
      continue;
    }
    textPositionsByKey.set(
      `${position.itemId}:${position.outputIndex}:${position.contentIndex ?? 0}`,
      position
    );
  }
  let hasAuthoritativeMessage = false;
  for (let outputIndex = 0; outputIndex < output.length; outputIndex++) {
    const item = output[outputIndex];
    if (item == null || typeof item !== 'object') {
      continue;
    }
    const itemRecord = item as Record<string, unknown>;
    rawOutputIndices.set(itemRecord, outputIndex);
    const itemKey = getResponsesReplayItemKey(itemRecord);
    if (itemKey != null) {
      authoritativeOutputIndicesByItemId.set(itemKey, outputIndex);
    }
    if (!('type' in item) || item.type !== 'message') {
      continue;
    }
    hasAuthoritativeMessage = true;
    const messageItemId =
      'id' in item && typeof item.id === 'string'
        ? item.id
        : `message-${outputIndex}`;
    messagePositionsByKey.set(`${messageItemId}:${outputIndex}`, {
      itemId: messageItemId,
      kind: 'message',
      outputIndex,
    });
    if (
      history != null ||
      !('content' in item) ||
      !Array.isArray(item.content)
    ) {
      continue;
    }
    for (
      let contentIndex = 0;
      contentIndex < item.content.length;
      contentIndex++
    ) {
      const content = item.content[contentIndex];
      if (
        content == null ||
        typeof content !== 'object' ||
        !('type' in content) ||
        content.type !== 'output_text' ||
        !('text' in content) ||
        typeof content.text !== 'string' ||
        content.text.length === 0
      ) {
        continue;
      }
      const itemId =
        'id' in item && typeof item.id === 'string'
          ? item.id
          : `message-${outputIndex}`;
      textPositionsByKey.set(`${itemId}:${outputIndex}:${contentIndex}`, {
        contentIndex,
        itemId,
        kind: 'text',
        outputIndex,
      });
    }
  }
  if (history?.source.coverage === 'complete-output') {
    for (const contribution of history.contributions) {
      if (contribution.kind !== 'text') {
        continue;
      }
      const itemId =
        contribution.itemId ?? `message-${contribution.outputIndex}`;
      textPositionsByKey.set(
        `${itemId}:${contribution.outputIndex}:${contribution.contentIndex ?? 0}`,
        {
          itemId,
          kind: 'text',
          outputIndex: contribution.outputIndex,
          contentIndex: contribution.contentIndex,
        }
      );
    }
  }
  if (hasAuthoritativeMessage) {
    for (const [itemId, outputIndex] of authoritativeOutputIndicesByItemId) {
      outputPositionsByItemId.set(itemId, {
        itemId,
        kind: 'output',
        outputIndex,
      });
    }
  }
  const textPositions = [...textPositionsByKey.values()].sort(
    (a, b) =>
      a.outputIndex - b.outputIndex ||
      (a.contentIndex ?? 0) - (b.contentIndex ?? 0)
  );
  const messagePositions = [...messagePositionsByKey.values()].sort(
    (a, b) => a.outputIndex - b.outputIndex
  );
  const textCountBeforeOutputIndex = new Map<number, number>();
  const positionedOutputIndexSet = new Set<number>();
  for (const position of outputPositionsByItemId.values()) {
    positionedOutputIndexSet.add(position.outputIndex);
  }
  if (hasAuthoritativeMessage) {
    for (const outputIndex of rawOutputIndices.values()) {
      positionedOutputIndexSet.add(outputIndex);
    }
  }
  const positionedOutputIndices = [...positionedOutputIndexSet].sort(
    (a, b) => a - b
  );
  let textPositionIndex = 0;
  for (const outputIndex of positionedOutputIndices) {
    while (
      textPositionIndex < textPositions.length &&
      textPositions[textPositionIndex].outputIndex < outputIndex
    ) {
      textPositionIndex++;
    }
    textCountBeforeOutputIndex.set(outputIndex, textPositionIndex);
  }
  const positionsByItemId = new Map<string, ResponsesReplayItemPosition>();
  for (const [itemId, position] of outputPositionsByItemId) {
    positionsByItemId.set(itemId, {
      outputIndex: position.outputIndex,
      textIndex: textCountBeforeOutputIndex.get(position.outputIndex) ?? 0,
    });
  }
  const getPosition = (
    item: Record<string, unknown>
  ): ResponsesReplayItemPosition => {
    const itemId = getResponsesReplayItemKey(item);
    const position = itemId != null ? positionsByItemId.get(itemId) : undefined;
    if (position == null) {
      const rawOutputIndex = rawOutputIndices.get(item);
      return {
        outputIndex: rawOutputIndex ?? Number.MAX_SAFE_INTEGER,
        ...(hasAuthoritativeMessage && rawOutputIndex != null
          ? {
            textIndex: textCountBeforeOutputIndex.get(rawOutputIndex) ?? 0,
          }
          : {}),
      };
    }
    return position;
  };
  const pushServerToolResult = (
    block: LangChainContentBlock.Standard | undefined,
    item: Record<string, unknown>,
    subIndex = 0
  ): void => {
    if (block == null) {
      return;
    }
    const itemId = getResponsesReplayItemKey(item);
    if (itemId != null) {
      positionedServerToolResultIds.add(itemId);
    }
    serverToolResults.push({ block, ...getPosition(item), subIndex });
  };
  for (const item of output) {
    if (item == null || typeof item !== 'object' || !('type' in item)) {
      continue;
    }
    if (item.type === 'code_interpreter_call') {
      if (
        !('outputs' in item) ||
        !Array.isArray(item.outputs) ||
        !('status' in item)
      ) {
        continue;
      }
      const resultStatus = item.status === 'completed' ? 'success' : 'error';
      const returnCode = item.status === 'completed' ? 0 : 1;
      for (
        let resultIndex = 0;
        resultIndex < item.outputs.length;
        resultIndex++
      ) {
        const result = item.outputs[resultIndex];
        if (
          result == null ||
          typeof result !== 'object' ||
          !('type' in result)
        ) {
          continue;
        }
        if (
          result.type === 'logs' &&
          'logs' in result &&
          typeof result.logs === 'string'
        ) {
          pushServerToolResult(
            createNeutralServerToolResult(
              {
                type: 'code_interpreter_output',
                returnCode,
                stdout: result.logs,
              },
              resultStatus,
              maxChars,
              'code_interpreter'
            ),
            item,
            resultIndex
          );
          continue;
        }
        if (
          result.type !== 'image' ||
          !('url' in result) ||
          typeof result.url !== 'string' ||
          result.url.length === 0
        ) {
          continue;
        }
        const resultUrl: string = result.url;
        if (
          replayProjection === 'native' &&
          resultUrl.startsWith('data:image/')
        ) {
          pushServerToolResult(
            {
              type: 'image',
              url: resultUrl,
              extras: createServerToolResultExtras('code_interpreter'),
            },
            item,
            resultIndex
          );
          continue;
        }
        const mediaResult = createNeutralServerToolResult(
          { type: 'code_interpreter_image', url: resultUrl },
          resultStatus,
          maxChars,
          'code_interpreter'
        );
        pushServerToolResult(mediaResult, item, resultIndex);
      }
      continue;
    }
    if (item.type === 'file_search_call') {
      const result = createNeutralServerToolResult(
        'results' in item && Array.isArray(item.results)
          ? { results: item.results }
          : undefined,
        'status' in item && item.status === 'completed' ? 'success' : 'error',
        maxChars,
        'file_search'
      );
      pushServerToolResult(result, item);
      continue;
    }
    if (item.type === 'web_search_call') {
      const result = createNeutralServerToolResult(
        {
          ...('action' in item ? { action: item.action } : {}),
          ...('results' in item && Array.isArray(item.results)
            ? { results: item.results }
            : {}),
        },
        'status' in item && item.status === 'completed' ? 'success' : 'error',
        maxChars,
        'web_search'
      );
      pushServerToolResult(result, item);
      continue;
    }
    if (item.type === 'tool_search_output') {
      const result = createNeutralServerToolResult(
        'tools' in item && Array.isArray(item.tools)
          ? { tools: item.tools }
          : undefined,
        'status' in item && item.status === 'completed' ? 'success' : 'error',
        maxChars,
        'tool_search'
      );
      pushServerToolResult(result, item);
      continue;
    }
    if (item.type === 'mcp_list_tools') {
      const hasError =
        'error' in item &&
        typeof item.error === 'string' &&
        item.error.length > 0;
      const listToolsResult = createNeutralServerToolResult(
        {
          ...('server_label' in item && typeof item.server_label === 'string'
            ? { serverLabel: item.server_label }
            : {}),
          ...('tools' in item && Array.isArray(item.tools)
            ? { tools: item.tools }
            : {}),
          ...(hasError ? { error: item.error } : {}),
        },
        hasError ? 'error' : 'success',
        maxChars,
        'mcp_list_tools'
      );
      pushServerToolResult(listToolsResult, item);
      continue;
    }
    if (item.type === 'mcp_call') {
      const hasOutput =
        'output' in item &&
        typeof item.output === 'string' &&
        item.output.length > 0;
      const hasError =
        'error' in item &&
        typeof item.error === 'string' &&
        item.error.length > 0;
      if (!hasOutput && !hasError) {
        continue;
      }
      const mcpOutput = {
        ...('name' in item && typeof item.name === 'string'
          ? { name: item.name }
          : {}),
        ...('server_label' in item && typeof item.server_label === 'string'
          ? { serverLabel: item.server_label }
          : {}),
        ...('output' in item && typeof item.output === 'string'
          ? { output: item.output }
          : {}),
        ...('error' in item && typeof item.error === 'string'
          ? { error: item.error }
          : {}),
      };
      const mcpResult = createNeutralServerToolResult(
        mcpOutput,
        hasError ||
          ('status' in item &&
            (item.status === 'failed' || item.status === 'incomplete'))
          ? 'error'
          : 'success',
        maxChars,
        'name' in item && typeof item.name === 'string' && item.name.length > 0
          ? item.name
          : 'mcp'
      );
      pushServerToolResult(mcpResult, item);
      continue;
    }
    if (
      item.type === 'local_shell_call_output' ||
      item.type === 'shell_call_output' ||
      item.type === 'apply_patch_call_output'
    ) {
      if (!('output' in item)) {
        continue;
      }
      const result = createNeutralServerToolResult(
        item.output,
        'status' in item &&
          (item.status === 'failed' || item.status === 'incomplete')
          ? 'error'
          : 'success',
        maxChars,
        RESPONSES_REPLAY_OUTPUT_TOOL_NAMES[item.type]
      );
      pushServerToolResult(result, item);
      continue;
    }
    if (item.type === 'program_output') {
      if (!('result' in item)) {
        continue;
      }
      const result = createNeutralServerToolResult(
        item.result,
        'status' in item && item.status === 'incomplete' ? 'error' : 'success',
        maxChars,
        'program'
      );
      pushServerToolResult(result, item);
      continue;
    }
    if (item.type !== 'image_generation_call') {
      continue;
    }
    if ('id' in item && typeof item.id === 'string' && item.id.length > 0) {
      ids.add(item.id);
    }
    if (
      !('result' in item) ||
      typeof item.result !== 'string' ||
      item.result.length === 0
    ) {
      continue;
    }
    data.add(item.result);
    if ('status' in item && item.status === 'completed') {
      const itemId =
        'id' in item && typeof item.id === 'string' && item.id.length > 0
          ? item.id
          : undefined;
      if (
        (itemId != null && emittedIds.has(itemId)) ||
        (itemId == null && emittedData.has(item.result))
      ) {
        continue;
      }
      if (itemId != null) {
        emittedIds.add(itemId);
      } else {
        emittedData.add(item.result);
      }
      blocks.push({
        block: {
          type: 'image',
          mimeType: getGeneratedImageMimeType(item.result),
          data: item.result,
          extras: createServerToolResultExtras('image_generation'),
        },
        ...getPosition(item),
        subIndex: 0,
      });
    }
  }
  return {
    generatedImages: {
      blocks: blocks.sort(comparePositionedResponsesReplayBlocks),
      data,
      ids,
    },
    messagePositions,
    positionedServerToolResultIds,
    positionsByItemId,
    serverToolResults: serverToolResults.sort(
      comparePositionedResponsesReplayBlocks
    ),
    textPositions,
  };
}

function getSelfContainedResponsesV0Images(
  message: AIMessage
): PositionedResponsesImages {
  if (!Array.isArray(message.content)) {
    return { blocks: [], textCount: 0 };
  }
  const images: PositionedResponsesImage[] = [];
  let textCount = 0;
  for (const block of message.content as LangChainContentBlock.Standard[]) {
    if (block.type === 'text') {
      if (isCompleteToolStreamContentBlock(block)) {
        textCount++;
      }
      continue;
    }
    if (
      block.type === 'image' &&
      (('fileId' in block &&
        typeof block.fileId === 'string' &&
        block.fileId.length > 0) ||
        ('url' in block &&
          typeof block.url === 'string' &&
          block.url.length > 0) ||
        ('data' in block &&
          ((typeof block.data === 'string' && block.data.length > 0) ||
            (block.data instanceof Uint8Array && block.data.length > 0))))
    ) {
      images.push({ block, textIndex: textCount });
    }
  }
  return { blocks: images, textCount };
}

function getResponsesV0ContentBlocks(
  message: AIMessage,
  authoritativeOutput: unknown[]
): AIMessage['content'] {
  let content = message.content;
  if (typeof content === 'string') {
    content = toLangChainContent(
      content.length > 0 ? [{ type: 'text', text: content }] : []
    );
  }
  const translationMessage = cloneAIMessageWithResponsesReplayState(
    message,
    content,
    message.id,
    {
      ...message.additional_kwargs,
      tool_outputs: authoritativeOutput,
    },
    {
      ...message.response_metadata,
      model_provider: 'openai',
    }
  );
  return toLangChainContent(translationMessage.contentBlocks);
}

function isPreemptedOpenAIResponsesMessage(message: AIMessage): boolean {
  const metadata = message.response_metadata;
  if (metadata.preempted !== true || metadata.model_provider !== 'openai') {
    return false;
  }
  if (
    message.id?.startsWith('msg_') === true ||
    message.id?.startsWith('resp_') === true ||
    (typeof metadata.id === 'string' && metadata.id.startsWith('resp_')) ||
    Array.isArray(metadata.output) ||
    metadata.tool_outputs != null ||
    Array.isArray(message.additional_kwargs.tool_outputs) ||
    message.additional_kwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY] != null ||
    message.additional_kwargs.__openai_function_call_ids__ != null ||
    message.additional_kwargs.__openai_custom_tool_call_ids__ != null ||
    (message.additional_kwargs.reasoning != null &&
      typeof message.additional_kwargs.reasoning === 'object')
  ) {
    return true;
  }
  return false;
}

/**
 * A sealed Responses turn cannot prove that provider-side item ids were
 * retained: the response object does not echo the request's `store` flag.
 * Project a provider-neutral clone while preserving self-contained encrypted
 * reasoning and the original graph/checkpoint message.
 */
function projectPreemptedOpenAIResponsesMessage(
  message: AIMessage,
  maxChars: number,
  replayProjection: ResponsesReplayProjection,
  preparation: ToolHistoryPreparation = createToolHistoryPreparation()
): AIMessage {
  if (!isPreemptedOpenAIResponsesMessage(message)) {
    return message;
  }

  const metadata = message.response_metadata;
  const additionalKwargs = { ...message.additional_kwargs };
  const retainsEncryptedReasoning =
    replayProjection === 'native' &&
    hasReplayableEncryptedReasoning(additionalKwargs.reasoning);
  const history = preparation.get(message);
  const authoritativeOutput =
    history?.source.items ?? getAuthoritativeResponsesOutput(message);
  const {
    generatedImages,
    messagePositions,
    positionedServerToolResultIds,
    positionsByItemId,
    serverToolResults,
    textPositions,
  } = getResponsesReplayArtifacts(
    authoritativeOutput,
    maxChars,
    replayProjection,
    additionalKwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY],
    history
  );
  const reasoningItemId =
    retainsEncryptedReasoning &&
    additionalKwargs.reasoning != null &&
    typeof additionalKwargs.reasoning === 'object' &&
    'id' in additionalKwargs.reasoning &&
    typeof additionalKwargs.reasoning.id === 'string'
      ? additionalKwargs.reasoning.id
      : undefined;
  const reasoningPosition =
    reasoningItemId != null
      ? positionsByItemId.get(reasoningItemId)
      : undefined;
  const isV1 = metadata.output_version === 'v1';
  const translatesV0 =
    !isV1 &&
    (replayProjection === 'fallback' || authoritativeOutput.length > 0);
  const originalImages =
    replayProjection === 'native' && translatesV0
      ? getSelfContainedResponsesV0Images(message)
      : undefined;
  let contentForProjection = message.content;
  if (!isV1 && translatesV0) {
    contentForProjection = getResponsesV0ContentBlocks(
      message,
      authoritativeOutput
    );
  }
  const projectedContent = projectPreemptedResponsesV1Content(
    contentForProjection,
    isV1 || translatesV0 ? additionalKwargs.reasoning : undefined,
    maxChars,
    replayProjection,
    replayProjection === 'native' ? generatedImages : undefined,
    originalImages,
    serverToolResults,
    positionedServerToolResultIds,
    reasoningPosition,
    textPositions,
    messagePositions
  );
  const promotesV0 =
    !isV1 &&
    (replayProjection === 'fallback' ||
      generatedImages.blocks.length > 0 ||
      projectedContent.preservedServerToolResult);
  const unpromotedV0Content =
    contentForProjection === message.content
      ? projectedContent.content
      : projectPreemptedResponsesV1Content(
        message.content,
        undefined,
        maxChars,
        replayProjection
      ).content;
  const content =
    isV1 || promotesV0 ? projectedContent.content : unpromotedV0Content;
  const hasUnsafeProviderReferences =
    content !== message.content ||
    message.id?.startsWith('msg_') === true ||
    (!retainsEncryptedReasoning && additionalKwargs.reasoning != null) ||
    additionalKwargs.tool_outputs != null ||
    additionalKwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY] != null ||
    additionalKwargs.__openai_function_call_ids__ != null ||
    additionalKwargs.__openai_custom_tool_call_ids__ != null ||
    metadata.id != null ||
    metadata.output != null ||
    metadata.tool_outputs != null;
  if (!hasUnsafeProviderReferences) {
    return message;
  }
  if (!retainsEncryptedReasoning) {
    delete additionalKwargs.reasoning;
  }
  delete additionalKwargs.tool_outputs;
  delete additionalKwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY];
  delete additionalKwargs[OPENAI_RESPONSES_ACTIVE_REASONING_ID_KEY];
  delete additionalKwargs.__openai_function_call_ids__;
  delete additionalKwargs.__openai_custom_tool_call_ids__;

  const responseMetadata: AIMessage['response_metadata'] = { ...metadata };
  delete responseMetadata.id;
  delete responseMetadata.output;
  delete responseMetadata.tool_outputs;
  if (promotesV0) {
    responseMetadata.output_version = 'v1';
  }

  return cloneAIMessageWithResponsesReplayState(
    message,
    content,
    message.id?.startsWith('msg_') === true ? undefined : message.id,
    additionalKwargs,
    responseMetadata
  );
}

/** Applies sealed-Responses safety and drops incomplete streamed text input. */
export function projectToolStreamContentForProvider(
  messages: BaseMessage[],
  responsesReplayProjection?: ResponsesReplayProjection,
  maxChars = HARD_MAX_TOOL_RESULT_CHARS,
  preparation: ToolHistoryPreparation = createToolHistoryPreparation()
): BaseMessage[] {
  let projected: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (message.getType() !== 'ai') {
      continue;
    }
    const assistantMessage = message as AIMessage;
    if (
      responsesReplayProjection != null &&
      isPreemptedOpenAIResponsesMessage(assistantMessage)
    ) {
      const replaySafeMessage = projectPreemptedOpenAIResponsesMessage(
        assistantMessage,
        maxChars,
        responsesReplayProjection,
        preparation
      );
      if (replaySafeMessage !== assistantMessage) {
        projected ??= [...messages];
        projected[i] = replaySafeMessage;
      }
      continue;
    }
    const content = dropIncompleteToolStreamContent(assistantMessage.content);
    if (content === assistantMessage.content) {
      continue;
    }
    projected ??= [...messages];
    projected[i] = cloneAIMessageWithContent(assistantMessage, content);
  }
  return projected ?? messages;
}

type CacheControlledTextProjection = 'serialize' | 'preserve' | 'text';

function projectStructuredOpenAIToolContent(
  content: ToolMessage['content'],
  maxChars: number,
  cacheControlledTextProjection: CacheControlledTextProjection
): ToolMessage['content'] {
  if (cacheControlledTextProjection !== 'serialize') {
    const cacheControlledContent = getBoundedCacheControlledTextToolContent(
      content,
      maxChars
    );
    if (cacheControlledContent != null) {
      return cacheControlledTextProjection === 'preserve'
        ? cacheControlledContent
        : cacheControlledContent[0].text;
    }
    const singleTextContent = getBoundedSingleTextToolContent(
      content,
      maxChars
    );
    if (singleTextContent != null) {
      return singleTextContent;
    }
  }
  const serializableContent =
    cacheControlledTextProjection === 'preserve'
      ? stripAnthropicCacheControl([{ content }])[0].content
      : content;
  return serializeToolContentBounded(serializableContent, maxChars);
}

/**
 * OpenAI Chat tool messages only accept strings or text-only parts, while the
 * Responses API serializes any structured ToolMessage after graph accounting.
 * Project every non-string tool result to one bounded string before the final
 * provider payload is measured so both APIs receive the exact representation
 * the budget guard counted. Native Responses computer screenshots stay
 * structured because their dedicated converter sends the media block directly.
 */
function projectOpenAIToolMessageContentInternal(
  messages: BaseMessage[],
  maxChars: number,
  nativeResponsesProjection: boolean,
  cacheControlledTextProjection: CacheControlledTextProjection
): BaseMessage[] {
  const pendingComputerCallIds: string[] = [];
  const seenComputerCallIds = new Set<string>();
  let projected: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    const messageRole = (message as BaseMessage & { role?: unknown }).role;
    const isAssistant =
      message.getType() === 'ai' || messageRole === 'assistant';
    if (isAssistant) {
      let assistantMessage = message as AIMessage;
      if (nativeResponsesProjection) {
        const replaySafeMessage = projectPreemptedOpenAIResponsesMessage(
          assistantMessage,
          maxChars,
          'native'
        );
        if (replaySafeMessage !== assistantMessage) {
          projected ??= [...messages];
          projected[i] = replaySafeMessage;
          assistantMessage = replaySafeMessage;
        }
      }
      const parsedComputerCallIds = new Set<string>();
      const toolCalls = assistantMessage.tool_calls;
      if (Array.isArray(toolCalls)) {
        for (const toolCall of toolCalls) {
          const record = toolCall as ToolCall & {
            isComputerTool?: unknown;
          };
          if (
            record.type !== 'tool_call' ||
            record.isComputerTool !== true ||
            typeof record.id !== 'string' ||
            record.id === ''
          ) {
            continue;
          }
          if (parsedComputerCallIds.has(record.id)) {
            throw new Error(`Duplicate computer call id "${record.id}"`);
          }
          parsedComputerCallIds.add(record.id);
          if (seenComputerCallIds.has(record.id)) {
            throw new Error(`Duplicate computer call id "${record.id}"`);
          }
          seenComputerCallIds.add(record.id);
          pendingComputerCallIds.push(record.id);
        }
      }

      const rawOutput = (
        assistantMessage.response_metadata as {
          output?: unknown;
        }
      ).output;
      const fallbackOutput = (
        assistantMessage.additional_kwargs as {
          tool_outputs?: unknown;
        }
      ).tool_outputs;
      let actualToolOutputs: unknown[] = [];
      if (Array.isArray(rawOutput) && rawOutput.length > 0) {
        actualToolOutputs = rawOutput;
      } else if (Array.isArray(fallbackOutput)) {
        actualToolOutputs = fallbackOutput;
      }
      const rawComputerCallIds = new Set<string>();
      for (const item of actualToolOutputs) {
        if (item == null || typeof item !== 'object') {
          continue;
        }
        const record = item as {
          type?: unknown;
          call_id?: unknown;
        };
        if (
          record.type !== 'computer_call' ||
          typeof record.call_id !== 'string' ||
          record.call_id === ''
        ) {
          continue;
        }
        if (rawComputerCallIds.has(record.call_id)) {
          throw new Error(`Duplicate computer call id "${record.call_id}"`);
        }
        rawComputerCallIds.add(record.call_id);
        // LangChain can retain the same call in parsed and raw forms. It sends
        // one logical call, so collapse that representation duplicate.
        if (parsedComputerCallIds.has(record.call_id)) {
          continue;
        }
        if (seenComputerCallIds.has(record.call_id)) {
          throw new Error(`Duplicate computer call id "${record.call_id}"`);
        }
        seenComputerCallIds.add(record.call_id);
        pendingComputerCallIds.push(record.call_id);
      }

      if (
        nativeResponsesProjection &&
        Array.isArray(toolCalls) &&
        rawComputerCallIds.size > 0
      ) {
        /**
         * The non-streaming Responses converter marks parsed computer calls,
         * but the streaming converter currently emits the same call as an
         * ordinary parsed `computer_use` tool call. In both cases the raw
         * `computer_call` item is authoritative and is replayed by LangChain,
         * so remove every parsed representation with the same call id.
         */
        const projectedToolCalls = toolCalls.filter(
          (toolCall) =>
            typeof toolCall.id !== 'string' ||
            !rawComputerCallIds.has(toolCall.id)
        );
        if (projectedToolCalls.length !== toolCalls.length) {
          projected ??= [...messages];
          projected[i] = cloneAIMessageWithToolCalls(
            assistantMessage,
            projectedToolCalls,
            rawComputerCallIds
          );
        }
      }
    }

    if (
      message instanceof ToolMessage &&
      hasComputerCallOutputMarker(message)
    ) {
      const screenshot = getComputerCallOutputScreenshot(message.content);
      if (screenshot == null) {
        throw new Error('Invalid computer call output screenshot');
      }
      if (pendingComputerCallIds[0] !== message.tool_call_id) {
        throw new Error(
          `Invalid computer call output pairing for "${message.tool_call_id}"`
        );
      }
      pendingComputerCallIds.shift();
      projected ??= [...messages];
      projected[i] = cloneToolMessageWithContent(message, [screenshot]);
      continue;
    }
    if (
      !(message instanceof ToolMessage) ||
      typeof message.content === 'string'
    ) {
      continue;
    }
    projected ??= [...messages];
    projected[i] = cloneToolMessageWithContent(
      message,
      projectStructuredOpenAIToolContent(
        message.content,
        maxChars,
        cacheControlledTextProjection
      )
    );
  }
  if (pendingComputerCallIds.length > 0) {
    throw new Error(
      `Missing computer call output for "${pendingComputerCallIds[0]}"`
    );
  }
  return projected ?? messages;
}

/** Projects OpenAI-compatible tool content without changing parsed call parents. */
export function projectOpenAIToolMessageContent(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  return projectOpenAIToolMessageContentInternal(
    messages,
    maxChars,
    false,
    'serialize'
  );
}

/** Projects an actual OpenAI-compatible Chat attempt and removes cache metadata. */
export function projectOpenAIChatToolMessageContent(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  return projectOpenAIToolMessageContentInternal(
    messages,
    maxChars,
    false,
    'text'
  );
}

/** Preserves OpenRouter's cache-decorated text blocks for a Chat attempt. */
export function projectOpenRouterToolMessageContent(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  return projectOpenAIToolMessageContentInternal(
    messages,
    maxChars,
    false,
    'preserve'
  );
}

/** Projects Responses tool content and collapses parsed/raw computer-call mirrors. */
export function projectOpenAIResponsesToolMessageContent(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  return projectOpenAIToolMessageContentInternal(
    messages,
    maxChars,
    true,
    'text'
  );
}

/** Removes Anthropic/OpenRouter cache metadata before unsupported providers run. */
export function projectCacheControlledToolOutputsToText(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  let projected: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (
      !(message instanceof ToolMessage) ||
      typeof message.content === 'string'
    ) {
      continue;
    }
    const cacheControlledContent = getBoundedCacheControlledTextToolContent(
      message.content,
      maxChars
    );
    if (cacheControlledContent == null) {
      continue;
    }
    projected ??= [...messages];
    projected[i] = cloneToolMessageWithContent(
      message,
      cacheControlledContent[0].text
    );
  }
  return projected ?? messages;
}

/** Unwraps a canonical single text block after provider cache markers are removed. */
export function projectSingleTextToolOutputsToText(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  let projected: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (
      !(message instanceof ToolMessage) ||
      typeof message.content === 'string'
    ) {
      continue;
    }
    const text = getBoundedSingleTextToolContent(message.content, maxChars);
    if (text == null) {
      continue;
    }
    projected ??= [...messages];
    projected[i] = cloneToolMessageWithContent(message, text);
  }
  return projected ?? messages;
}

/** Serializes provider-neutral structured tool outputs without media pairing. */
export function projectStructuredToolOutputsToText(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  let projected: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (
      !(message instanceof ToolMessage) ||
      typeof message.content === 'string' ||
      hasComputerCallOutputMarker(message)
    ) {
      continue;
    }
    projected ??= [...messages];
    projected[i] = cloneToolMessageWithContent(
      message,
      serializeToolContentBounded(message.content, maxChars)
    );
  }
  return projected ?? messages;
}

/**
 * Non-Responses providers cannot consume native computer screenshots. Keep
 * the tool-call structure intact, but replace screenshot bytes with a bounded
 * text marker at the actual invocation boundary.
 */
export function projectComputerCallOutputsToText(
  messages: BaseMessage[]
): BaseMessage[] {
  let projected: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (
      !(message instanceof ToolMessage) ||
      !hasComputerCallOutputMarker(message)
    ) {
      continue;
    }
    projected ??= [...messages];
    projected[i] = stampSyntheticProviderMessage(
      cloneToolMessageWithContent(
        message,
        '[Computer screenshot omitted for this provider]'
      )
    );
  }
  return projected ?? messages;
}

export function projectAnthropicArtifactContent(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS
): BaseMessage[] {
  const lastMessage = messages[messages.length - 1];
  if (!(lastMessage instanceof ToolMessage)) return messages;

  // Find the latest AIMessage with tool_calls that this tool message belongs to
  const latestAIParentIndex = findLastIndex(
    messages,
    (msg) =>
      ((msg instanceof AIMessage || msg instanceof AIMessageChunk) &&
        (msg.tool_calls?.length ?? 0) > 0 &&
        msg.tool_calls?.some((tc) => tc.id === lastMessage.tool_call_id)) ??
      false
  );

  if (latestAIParentIndex === -1) return messages;

  // Build tool call ID set and merge artifact content in a single forward pass.
  const message = messages[latestAIParentIndex] as AIMessage | AIMessageChunk;
  const toolCallIdSet = new Set<string>();
  if (message.tool_calls) {
    for (const tc of message.tool_calls) {
      if (tc.id != null) {
        toolCallIdSet.add(tc.id);
      }
    }
  }

  let formattedMessages: BaseMessage[] | undefined;
  for (let j = latestAIParentIndex + 1; j < messages.length; j++) {
    const msg = messages[j];
    if (
      msg instanceof ToolMessage &&
      !isComputerCallOutputMessage(msg) &&
      toolCallIdSet.has(msg.tool_call_id) &&
      msg.artifact != null &&
      ((typeof msg.artifact?.content === 'string' &&
        msg.artifact.content.length > 0) ||
        (Array.isArray(msg.artifact?.content) &&
          msg.artifact.content.length > 0))
    ) {
      const artifactContent =
        typeof msg.artifact.content === 'string'
          ? [
            {
              type: ContentTypes.TEXT,
              text: msg.artifact.content,
            },
          ]
          : msg.artifact.content;
      const baseContent = Array.isArray(msg.content)
        ? msg.content
        : stringifyToolMessageContent(msg.content);
      const aggregate = createBoundedContentAccumulator(maxChars);
      let contentProjection: ReturnType<
        typeof appendBoundedContentContribution
      > = { contributed: false, complete: true };
      /** Preserve the established empty-result text block shape without
       * crediting it as a provider-visible source contribution. */
      if (baseContent === '') {
        aggregate.blocks.push({ type: ContentTypes.TEXT, text: '' });
        aggregate.hasContent = true;
      } else {
        contentProjection = appendBoundedContentContribution(
          aggregate,
          baseContent
        );
      }
      const artifactProjection = appendBoundedContentContribution(
        aggregate,
        artifactContent
      );
      const content = toLangChainContent(aggregate.blocks);
      formattedMessages ??= [...messages];
      const projectedMessage = cloneToolMessageWithContent(msg, content, {
        ...msg.artifact,
        content: [],
      });
      const toolProvenanceParts = projectProviderMessageAttribution(
        msg,
        'tool'
      );
      const projectedProvenanceParts: ProviderMessageProvenancePart[] = [];
      if (contentProjection.contributed) {
        const retainedParts = retainProjectedContentPartProvenance(
          msg,
          toolProvenanceParts,
          contentProjection.retainedContentPartIndices,
          contentProjection.complete
        );
        for (const part of retainedParts) {
          projectedProvenanceParts.push(part);
        }
      }
      if (artifactProjection.contributed) {
        const artifactPart = projectUnindexedProviderMessageAttribution(
          msg,
          'tool'
        );
        const previousPart =
          projectedProvenanceParts[projectedProvenanceParts.length - 1];
        if (
          projectedProvenanceParts.length === 0 ||
          previousPart.attribution !== artifactPart.attribution ||
          previousPart.sourceMessageId !== artifactPart.sourceMessageId ||
          previousPart.sourceContentPartIndices != null
        ) {
          projectedProvenanceParts.push(artifactPart);
        }
      }
      if (projectedProvenanceParts.length > 0) {
        setProviderMessageProvenance(
          projectedMessage,
          projectedProvenanceParts
        );
      }
      formattedMessages[j] = projectedMessage;
    }
  }
  return formattedMessages ?? messages;
}

function projectProviderMessageAttribution(
  message: BaseMessage,
  attribution: ProviderMessageProvenancePart['attribution']
): ProviderMessageProvenancePart[] {
  const explicit = getProviderMessageProvenance(message);
  const sourceMessageIds = getProviderSourceMessageIds(message);
  const explicitSourceIds = new Set<string>();
  for (const part of explicit?.parts ?? []) {
    if (part.sourceMessageId != null) {
      explicitSourceIds.add(part.sourceMessageId);
    }
  }
  const soleSourceFallback =
    explicitSourceIds.size === 0 && sourceMessageIds.length === 1
      ? sourceMessageIds[0]
      : undefined;
  const parts: ProviderMessageProvenancePart[] = [];
  const representedSourceIds = new Set<string>();
  for (const part of explicit?.parts ?? []) {
    const sourceMessageId = part.sourceMessageId ?? soleSourceFallback;
    if (sourceMessageId != null) {
      representedSourceIds.add(sourceMessageId);
    }
    parts.push({
      attribution,
      ...(sourceMessageId != null && { sourceMessageId }),
      ...(part.sourceContentPartIndices != null && {
        sourceContentPartIndices: part.sourceContentPartIndices,
      }),
    });
  }
  for (const sourceMessageId of sourceMessageIds) {
    if (!representedSourceIds.has(sourceMessageId)) {
      parts.push({ attribution, sourceMessageId });
    }
  }
  return parts.length > 0 ? parts : [{ attribution }];
}

/** Artifacts are attached to a tool message rather than one of its content
 * positions. Retain the source row only when it is unambiguous, but never
 * claim a raw content-part index for artifact-only bytes. */
function projectUnindexedProviderMessageAttribution(
  message: BaseMessage,
  attribution: ProviderMessageProvenancePart['attribution']
): ProviderMessageProvenancePart {
  const sourceMessageIds = getProviderSourceMessageIds(message);
  return {
    attribution,
    ...(sourceMessageIds.length === 1 && {
      sourceMessageId: sourceMessageIds[0],
    }),
  };
}

/**
 * Mutating compatibility wrapper retained for existing package consumers.
 * New provider-call paths should use `projectAnthropicArtifactContent`.
 */
export function formatAnthropicArtifactContent(messages: BaseMessage[]): void {
  const projected = projectAnthropicArtifactContent(messages);
  if (projected === messages) {
    return;
  }
  for (let i = 0; i < messages.length; i++) {
    if (
      messages[i] instanceof ToolMessage &&
      projected[i] instanceof ToolMessage &&
      projected[i] !== messages[i]
    ) {
      messages[i].content = projected[i].content;
      const provenance = getProviderMessageProvenance(projected[i]);
      if (provenance != null) {
        setProviderMessageProvenance(messages[i], provenance.parts);
      }
    }
  }
}

export function projectArtifactPayload(
  messages: BaseMessage[],
  maxChars = HARD_MAX_TOOL_RESULT_CHARS,
  replayImages = true
): BaseMessage[] {
  const lastMessageY = messages[messages.length - 1];
  if (!(lastMessageY instanceof ToolMessage)) return messages;

  // Find the latest AIMessage with tool_calls that this tool message belongs to
  const latestAIParentIndex = findLastIndex(
    messages,
    (msg) =>
      ((msg instanceof AIMessage || msg instanceof AIMessageChunk) &&
        (msg.tool_calls?.length ?? 0) > 0 &&
        msg.tool_calls?.some((tc) => tc.id === lastMessageY.tool_call_id)) ??
      false
  );

  if (latestAIParentIndex === -1) return messages;

  // Single pass: collect relevant tool messages with artifacts and aggregate
  const aggregate = createBoundedContentAccumulator(maxChars);
  let formattedMessages: BaseMessage[] | undefined;
  const artifactProvenanceParts: ProviderMessageProvenancePart[] = [];

  for (let i = latestAIParentIndex + 1; i < messages.length; i++) {
    const msg = messages[i];
    if (
      !(msg instanceof ToolMessage) ||
      isComputerCallOutputMessage(msg) ||
      !(
        (typeof msg.artifact?.content === 'string' &&
          msg.artifact.content.length > 0) ||
        (Array.isArray(msg.artifact?.content) &&
          msg.artifact.content.length > 0)
      )
    ) {
      continue;
    }
    const contentProjection = appendBoundedContentContribution(
      aggregate,
      msg.content
    );
    formattedMessages ??= [...messages];
    const placeholder = cloneToolMessageWithContent(
      msg,
      'Tool response is included in the next message as a Human message',
      {
        ...msg.artifact,
        content: [],
      }
    );
    const toolProvenanceParts = projectProviderMessageAttribution(msg, 'tool');
    setProviderMessageProvenance(
      placeholder,
      toolProvenanceParts.map((part) => ({
        ...part,
        attribution: 'synthetic',
      }))
    );
    formattedMessages[i] = placeholder;
    const artifactProjection = appendBoundedContentContribution(
      aggregate,
      msg.artifact.content
    );
    if (contentProjection.contributed) {
      const retainedParts = retainProjectedContentPartProvenance(
        msg,
        toolProvenanceParts,
        contentProjection.retainedContentPartIndices,
        contentProjection.complete
      );
      for (const part of retainedParts) {
        artifactProvenanceParts.push(part);
      }
    }
    if (artifactProjection.contributed) {
      const artifactPart = projectUnindexedProviderMessageAttribution(
        msg,
        'tool'
      );
      const previousPart =
        artifactProvenanceParts[artifactProvenanceParts.length - 1];
      if (
        artifactProvenanceParts.length === 0 ||
        previousPart.attribution !== artifactPart.attribution ||
        previousPart.sourceMessageId !== artifactPart.sourceMessageId ||
        previousPart.sourceContentPartIndices != null
      ) {
        artifactProvenanceParts.push(artifactPart);
      }
    }
  }

  if (aggregate.hasContent && artifactProvenanceParts.length > 0) {
    const artifactPayload = new HumanMessage({
      content: toLangChainContent(aggregate.blocks),
    });
    setProviderMessageProvenance(artifactPayload, artifactProvenanceParts);
    formattedMessages?.push(artifactPayload);
  }
  const projected = formattedMessages ?? messages;
  return replayImages
    ? projectEarlierToolImages(
      messages,
      projected,
      latestAIParentIndex,
      aggregate.remainingChars
    )
    : projected;
}

/** Replay only images from completed tool calls in the current user turn.
 * Newest images consume the remaining artifact budget first. The projection
 * never writes image bytes back into graph state or persisted messages. */
function projectEarlierToolImages(
  source: BaseMessage[],
  projected: BaseMessage[],
  end: number,
  maxChars: number
): BaseMessage[] {
  if (maxChars <= 0) {
    return projected;
  }
  const eligible = new Set<number>();
  let pending = new Set<string>();
  for (let i = 0; i < end; i++) {
    const message = source[i];
    if (message instanceof HumanMessage) {
      eligible.clear();
      pending.clear();
    } else if (
      message instanceof AIMessage ||
      message instanceof AIMessageChunk
    ) {
      pending = new Set(
        message.tool_calls?.flatMap((call) =>
          call.id == null ? [] : [call.id]
        )
      );
    } else if (
      message instanceof ToolMessage &&
      pending.delete(message.tool_call_id)
    ) {
      eligible.add(i);
    }
  }
  if (eligible.size === 0) {
    return projected;
  }
  const aggregate = createBoundedContentAccumulator(maxChars);
  const payloads: HumanMessage[] = [];
  for (let i = end - 1; i >= 0 && aggregate.remainingChars > 0; i--) {
    const message = source[i];
    if (
      !eligible.has(i) ||
      !(message instanceof ToolMessage) ||
      isComputerCallOutputMessage(message) ||
      !Array.isArray(message.artifact?.content)
    ) {
      continue;
    }
    const images: t.MessageContentComplex[] = [];
    for (const block of message.artifact.content) {
      const image = getComputerCallOutputScreenshot([block]);
      if (image == null || !('image_url' in image)) {
        continue;
      }
      const normalized = {
        type: 'image_url',
        image_url: {
          url: image.image_url,
          ...('detail' in image &&
            image.detail != null && { detail: image.detail }),
        },
      };
      const chars = getToolContentCharLength([normalized]) + 1;
      if (chars > aggregate.remainingChars) {
        continue;
      }
      images.push(normalized);
      aggregate.remainingChars -= chars;
    }
    if (images.length === 0) {
      continue;
    }
    const payload = new HumanMessage({ content: toLangChainContent(images) });
    setProviderMessageProvenance(payload, [
      projectUnindexedProviderMessageAttribution(message, 'tool'),
    ]);
    payloads.push(payload);
  }
  return payloads.length === 0
    ? projected
    : [
      ...projected.slice(0, source.length),
      ...payloads.reverse(),
      ...projected.slice(source.length),
    ];
}

/**
 * Mutating compatibility wrapper retained for existing package consumers.
 * New provider-call paths should use `projectArtifactPayload`.
 */
export function formatArtifactPayload(messages: BaseMessage[]): void {
  const originalLength = messages.length;
  const projected = projectArtifactPayload(messages);
  if (projected === messages) {
    return;
  }
  for (let i = 0; i < originalLength; i++) {
    if (
      messages[i] instanceof ToolMessage &&
      projected[i] instanceof ToolMessage &&
      projected[i] !== messages[i]
    ) {
      messages[i].content = projected[i].content;
      const provenance = getProviderMessageProvenance(projected[i]);
      if (provenance != null) {
        setProviderMessageProvenance(messages[i], provenance.parts);
      }
    }
  }
  for (let i = originalLength; i < projected.length; i++) {
    messages.push(projected[i]);
  }
}

export function findLastIndex<T>(
  array: T[],
  predicate: (value: T) => boolean
): number {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i])) {
      return i;
    }
  }
  return -1;
}

/** Reads an own data property without invoking accessors; `undefined` otherwise. */
function readOwnDataProperty(target: object, key: PropertyKey): unknown {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(target, key);
    return descriptor != null && 'value' in descriptor
      ? descriptor.value
      : undefined;
  } catch {
    return undefined;
  }
}

/** Whether message content carries non-whitespace text in any text block. */
export function hasNonEmptyTextContent(
  content: BaseMessage['content']
): boolean {
  if (typeof content === 'string') {
    return content.trim() !== '';
  }
  if (!Array.isArray(content) || isProxy(content)) {
    return false;
  }
  /** Index reads through own data descriptors: `for...of` would invoke a
   * hostile `Symbol.iterator` or an accessor-backed element before any guard. */
  const length = readOwnDataProperty(content, 'length');
  const blockCount = typeof length === 'number' ? length : 0;
  for (let i = 0; i < blockCount; i++) {
    const block = readOwnDataProperty(content, i);
    if (typeof block !== 'object' || block == null || isProxy(block)) {
      continue;
    }
    try {
      const type = Object.getOwnPropertyDescriptor(block, 'type');
      if (
        type == null ||
        !('value' in type) ||
        (type.value !== ContentTypes.TEXT && type.value !== 'output_text')
      ) {
        continue;
      }
      const text = Object.getOwnPropertyDescriptor(block, ContentTypes.TEXT);
      if (
        text != null &&
        'value' in text &&
        typeof text.value === 'string' &&
        text.value.trim() !== ''
      ) {
        return true;
      }
    } catch {
      continue;
    }
  }
  return false;
}

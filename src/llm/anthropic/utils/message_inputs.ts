/* eslint-disable @typescript-eslint/explicit-function-return-type */
/* eslint-disable no-console */
/**
 * This util file contains functions for converting LangChain messages to Anthropic messages.
 */
import { createHash } from 'node:crypto';
import { isProxy } from 'node:util/types';
import { ToolCall } from '@langchain/core/messages/tool';
import {
  type BaseMessage,
  type SystemMessage,
  HumanMessage,
  type AIMessage,
  ToolMessage,
  isAIMessage,
  type Data,
  type StandardContentBlockConverter,
  MessageContentComplex,
  isDataContentBlock,
  convertToProviderContentBlock,
  parseBase64DataUrl,
} from '@langchain/core/messages';
import {
  AnthropicImageBlockParam,
  AnthropicMessageCreateParams,
  AnthropicTextBlockParam,
  AnthropicDocumentBlockParam,
  AnthropicThinkingBlockParam,
  AnthropicRedactedThinkingBlockParam,
  AnthropicServerToolUseBlockParam,
  AnthropicWebSearchToolResultBlockParam,
  isAnthropicImageBlockParam,
  AnthropicSearchResultBlockParam,
  AnthropicCompactionBlockParam,
  AnthropicToolResponse,
} from '../types';
import {
  cloneToolMessageWithContent,
  serializeStructuredValueBounded,
} from '@/utils/toolContent';
import { isReasoningContentBlock } from '@/messages/reasoningTypes';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { Constants } from '@/common';

type StandardTextBlock = Data.StandardTextBlock;
type StandardImageBlock = Data.StandardImageBlock;
type StandardFileBlock = Data.StandardFileBlock;
type ImageUrlContentBlock = MessageContentComplex & {
  image_url: string | { url: string };
};
type GoogleFunctionCallBlock = MessageContentComplex & {
  functionCall: {
    name: string;
    args: Record<string, unknown>;
  };
};

const ANTHROPIC_EMPTY_TEXT_PLACEHOLDER = '_';
/**
 * Name-first Claude ids (`claude-sonnet-4-6`, `claude-haiku-5.5`, Bedrock and
 * Vertex forms included). The minor is one or two digits so an 8-digit release
 * date (`claude-opus-4-20250514`) reads as major only. Number-first ids
 * (`claude-3-5-sonnet`) predate 4.6 and never match.
 */
const CLAUDE_FAMILY_VERSION_PATTERN =
  /claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:[-.](\d{1,2}))?(?!\d)/i;

function _formatImage(imageUrl: string) {
  const parsed = parseBase64DataUrl({ dataUrl: imageUrl });
  if (parsed) {
    return {
      type: 'base64',
      media_type: parsed.mime_type,
      data: parsed.data,
    };
  }
  let parsedUrl: URL;

  try {
    parsedUrl = new URL(imageUrl);
  } catch {
    throw new Error(
      [
        `Malformed image URL: ${JSON.stringify(
          imageUrl
        )}. Content blocks of type 'image_url' must be a valid http, https, or base64-encoded data URL.`,
        'Example: data:image/png;base64,/9j/4AAQSk...',
        'Example: https://example.com/image.jpg',
      ].join('\n\n')
    );
  }

  if (parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:') {
    return {
      type: 'url',
      url: imageUrl,
    };
  }

  throw new Error(
    [
      `Invalid image URL protocol: ${JSON.stringify(
        parsedUrl.protocol
      )}. Anthropic only supports images as http, https, or base64-encoded data URLs on 'image_url' content blocks.`,
      'Example: data:image/png;base64,/9j/4AAQSk...',
      'Example: https://example.com/image.jpg',
    ].join('\n\n')
  );
}

const ANTHROPIC_TOOL_USE_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;
const ANTHROPIC_TOOL_USE_ID_MAX_LENGTH = 64;
const ANTHROPIC_TOOL_USE_ID_HASH_LENGTH = 10;

/**
 * Normalize a tool-call ID to satisfy Anthropic's `^[a-zA-Z0-9_-]+$` and 64-char
 * constraints. Pure and deterministic — same input always yields the same output,
 * so paired `tool_use.id` and `tool_result.tool_use_id` stay matched without
 * needing a session map. IDs that already comply pass through unchanged.
 *
 * For non-compliant inputs we sanitize then append a short SHA-256 prefix of
 * the original ID to preserve uniqueness when truncation would otherwise
 * collapse distinct IDs to the same value (e.g. two long Responses-style IDs
 * sharing a 64-char prefix). The hash is computed against the raw input so
 * inputs that differ only after the truncation cutoff still produce distinct
 * outputs.
 */
export function normalizeAnthropicToolCallId(id: string): string;
export function normalizeAnthropicToolCallId(
  id: string | undefined
): string | undefined;
export function normalizeAnthropicToolCallId(
  id: string | undefined
): string | undefined {
  if (id == null) {
    return id;
  }
  if (
    id.length <= ANTHROPIC_TOOL_USE_ID_MAX_LENGTH &&
    ANTHROPIC_TOOL_USE_ID_PATTERN.test(id)
  ) {
    return id;
  }
  const sanitized = id.replace(/[^a-zA-Z0-9_-]/g, '_');
  const hash = createHash('sha256')
    .update(id)
    .digest('hex')
    .slice(0, ANTHROPIC_TOOL_USE_ID_HASH_LENGTH);
  const prefixMaxLength =
    ANTHROPIC_TOOL_USE_ID_MAX_LENGTH - ANTHROPIC_TOOL_USE_ID_HASH_LENGTH - 1;
  return `${sanitized.slice(0, prefixMaxLength)}_${hash}`;
}

/**
 * Lift any `cache_control` off the inner blocks of a tool result onto the
 * `tool_result` block itself. Anthropic documents the top-level
 * `messages.content` block as the cacheable position and does not document
 * caching of sub-content blocks; the API currently honors a nested marker, but
 * anchoring on the documented position keeps the single tail breakpoint robust
 * (and mirrors the Bedrock cachePoint hoist). The first marker found wins; it is
 * stripped from every inner block so exactly one survives, on the outer block.
 */
function hoistToolResultCacheControl(
  content: string | MessageContentComplex[]
): { content: string | MessageContentComplex[]; cacheControl: unknown } {
  if (!Array.isArray(content)) {
    return { content, cacheControl: undefined };
  }
  let cacheControl: unknown;
  const stripped = content.map((block) => {
    if (typeof block !== 'object' || block == null || isProxy(block)) {
      return block;
    }
    try {
      const descriptors = Object.getOwnPropertyDescriptors(block);
      const cacheControlDescriptor = descriptors.cache_control;
      if (
        cacheControlDescriptor.enumerable === true &&
        'value' in cacheControlDescriptor
      ) {
        cacheControl ??= cacheControlDescriptor.value;
        delete descriptors.cache_control;
        return Object.create(
          Object.getPrototypeOf(block),
          descriptors
        ) as MessageContentComplex;
      }
    } catch {
      return block;
    }
    return block;
  });
  // `stripped` is element-equal to `content` when no marker was present.
  return { content: stripped, cacheControl };
}

/**
 * A ToolMessage can already contain an Anthropic-native `tool_result` block
 * (for example, when a tool returns provider-native structured content). The
 * outer ToolMessage conversion supplies the one required result envelope, so
 * unwrap an exact single nested envelope rather than sending an invalid
 * `tool_result.content: [{ type: "tool_result", ... }]` payload.
 */
function unwrapToolMessageResultContent(content: ToolMessage['content']): {
  content: string | MessageContentComplex[] | undefined;
  cacheControl: unknown;
  isError: boolean | undefined;
} {
  if (!Array.isArray(content) || content.length !== 1) {
    return { content, cacheControl: undefined, isError: undefined };
  }

  const block = content[0];
  if (typeof block !== 'object' || block == null || isProxy(block)) {
    return { content, cacheControl: undefined, isError: undefined };
  }

  try {
    const type = Object.getOwnPropertyDescriptor(block, 'type');
    const nestedContent = Object.getOwnPropertyDescriptor(block, 'content');
    if (
      type?.enumerable !== true ||
      !('value' in type) ||
      type.value !== 'tool_result' ||
      (nestedContent != null &&
        (nestedContent.enumerable !== true || !('value' in nestedContent)))
    ) {
      return { content, cacheControl: undefined, isError: undefined };
    }

    const cacheControl = Object.getOwnPropertyDescriptor(
      block,
      'cache_control'
    );
    const isError = Object.getOwnPropertyDescriptor(block, 'is_error');
    const nested =
      nestedContent != null && 'value' in nestedContent
        ? nestedContent.value
        : undefined;
    if (typeof nested === 'string') {
      return {
        content: nested,
        cacheControl:
          cacheControl?.enumerable === true && 'value' in cacheControl
            ? cacheControl.value
            : undefined,
        isError:
          isError?.enumerable === true &&
          'value' in isError &&
          typeof isError.value === 'boolean'
            ? isError.value
            : undefined,
      };
    }
    if (Array.isArray(nested)) {
      return {
        content: nested as MessageContentComplex[],
        cacheControl:
          cacheControl?.enumerable === true && 'value' in cacheControl
            ? cacheControl.value
            : undefined,
        isError:
          isError?.enumerable === true &&
          'value' in isError &&
          typeof isError.value === 'boolean'
            ? isError.value
            : undefined,
      };
    }
    if (nested == null) {
      return {
        content: undefined,
        cacheControl:
          cacheControl?.enumerable === true && 'value' in cacheControl
            ? cacheControl.value
            : undefined,
        isError:
          isError?.enumerable === true &&
          'value' in isError &&
          typeof isError.value === 'boolean'
            ? isError.value
            : undefined,
      };
    }
    return {
      content: serializeStructuredValueBounded(
        nested,
        HARD_MAX_TOOL_RESULT_CHARS
      ).content,
      cacheControl:
        cacheControl?.enumerable === true && 'value' in cacheControl
          ? cacheControl.value
          : undefined,
      isError:
        isError?.enumerable === true &&
        'value' in isError &&
        typeof isError.value === 'boolean'
          ? isError.value
          : undefined,
    };
  } catch {
    return { content, cacheControl: undefined, isError: undefined };
  }
}

/**
 * A lone plain text block is the same tool result as its string, which is
 * how restored history carries it; the tail cache marker turns a string into
 * that block, so collapsing it keeps the bytes stable when the marker moves.
 */
function toPlainTextResult(
  content: string | MessageContentComplex[]
): string | MessageContentComplex[] {
  if (!Array.isArray(content) || content.length !== 1) {
    return content;
  }
  const [block] = content;
  const isPlainText =
    typeof block === 'object' &&
    block.type === 'text' &&
    typeof block.text === 'string' &&
    block.text !== '' &&
    Object.keys(block).every((key) => key === 'type' || key === 'text');
  return isPlainText ? (block.text as string) : content;
}

function _ensureMessageContents(
  messages: BaseMessage[]
): (SystemMessage | HumanMessage | AIMessage)[] {
  // Merge runs of human/tool messages into single human messages with content blocks.
  const updatedMsgs: BaseMessage[] = [];
  for (const message of messages) {
    if (message._getType() === 'tool') {
      if (typeof message.content === 'string') {
        const previousMessage = updatedMsgs[updatedMsgs.length - 1];
        if (
          previousMessage._getType() === 'human' &&
          Array.isArray(previousMessage.content) &&
          'type' in previousMessage.content[0] &&
          previousMessage.content[0].type === 'tool_result'
        ) {
          // If the previous message was a tool result, we merge this tool message into it.
          (previousMessage.content as MessageContentComplex[]).push({
            type: 'tool_result',
            content: message.content,
            tool_use_id: normalizeAnthropicToolCallId(
              (message as ToolMessage).tool_call_id
            ),
          });
        } else {
          // If not, we create a new human message with the tool result.
          updatedMsgs.push(
            new HumanMessage({
              content: [
                {
                  type: 'tool_result',
                  content: message.content,
                  tool_use_id: normalizeAnthropicToolCallId(
                    (message as ToolMessage).tool_call_id
                  ),
                },
              ],
            })
          );
        }
      } else {
        const toolMessage = message as ToolMessage;
        const toolMessageContent = toolMessage.content;
        const unwrapped = unwrapToolMessageResultContent(toolMessageContent);
        const formattedContent =
          unwrapped.content == null
            ? undefined
            : _formatContent(
              unwrapped.content === toolMessageContent
                ? toolMessage
                : cloneToolMessageWithContent(
                  toolMessage,
                      unwrapped.content as Parameters<
                        typeof cloneToolMessageWithContent
                      >[1]
                )
            );
        // Hoist a tail cache_control off the inner content onto the
        // tool_result block itself (the documented cacheable position).
        const { content: hoistedContent, cacheControl: innerCacheControl } =
          formattedContent != null
            ? hoistToolResultCacheControl(formattedContent)
            : { content: undefined, cacheControl: undefined };
        const cacheControl = unwrapped.cacheControl ?? innerCacheControl;
        const resultContent =
          hoistedContent != null ? toPlainTextResult(hoistedContent) : undefined;
        updatedMsgs.push(
          new HumanMessage({
            content: [
              {
                type: 'tool_result',
                ...(resultContent != null ? { content: resultContent } : {}),
                ...(unwrapped.isError != null
                  ? { is_error: unwrapped.isError }
                  : {}),
                ...(cacheControl != null
                  ? { cache_control: cacheControl as { type: 'ephemeral' } }
                  : {}),
                tool_use_id: normalizeAnthropicToolCallId(
                  (message as ToolMessage).tool_call_id
                ),
              },
            ],
          })
        );
      }
    } else {
      updatedMsgs.push(message);
    }
  }
  return updatedMsgs as (SystemMessage | HumanMessage | AIMessage)[];
}

/**
 * Anthropic requires `tool_use.input` to be a JSON object; anything else is
 * rejected with a 400 (`tool_use.input: Input should be an object`).
 * History can carry non-object inputs: streaming leaves the raw partial-JSON
 * string on the content block, and upstream context projections can leave
 * `null` (observed live on a summarization-retained tool call replayed after
 * compaction). A string is parsed when it forms complete JSON; every other
 * shape degrades to `{}` — the call already executed, so the replayed input
 * is informational and an empty object is the safe representation.
 */
/** True for `{}` — no own enumerable keys on a non-array, non-null object. */
function isEmptyPlainObject(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.keys(value).length === 0
  );
}

export function coerceAnthropicToolUseInput(
  input: unknown
): Record<string, unknown> {
  let candidate: unknown = input;
  if (typeof candidate === 'string') {
    try {
      candidate = JSON.parse(candidate);
    } catch {
      return {};
    }
  }
  return typeof candidate === 'object' &&
    candidate !== null &&
    !Array.isArray(candidate)
    ? (candidate as Record<string, unknown>)
    : {};
}

export function _convertLangChainToolCallToAnthropic(
  toolCall: ToolCall
): AnthropicToolResponse {
  if (toolCall.id === undefined) {
    throw new Error('Anthropic requires all tool calls to have an "id".');
  }
  const isServerTool = toolCall.id.startsWith(
    Constants.ANTHROPIC_SERVER_TOOL_PREFIX
  );
  return {
    type: isServerTool ? 'server_tool_use' : 'tool_use',
    id: isServerTool ? toolCall.id : normalizeAnthropicToolCallId(toolCall.id),
    name: toolCall.name,
    input: coerceAnthropicToolUseInput(toolCall.args),
  };
}

const standardContentBlockConverter: StandardContentBlockConverter<{
  text: AnthropicTextBlockParam;
  image: AnthropicImageBlockParam;
  file: AnthropicDocumentBlockParam;
}> = {
  providerName: 'anthropic',

  fromStandardTextBlock(block: StandardTextBlock): AnthropicTextBlockParam {
    return {
      type: 'text',
      text: block.text,
      ...('citations' in (block.metadata ?? {})
        ? { citations: block.metadata!.citations }
        : {}),
      ...('cache_control' in (block.metadata ?? {})
        ? { cache_control: block.metadata!.cache_control }
        : {}),
    } as AnthropicTextBlockParam;
  },

  fromStandardImageBlock(block: StandardImageBlock): AnthropicImageBlockParam {
    if (block.source_type === 'url') {
      const data = parseBase64DataUrl({
        dataUrl: block.url,
        asTypedArray: false,
      });
      if (data) {
        return {
          type: 'image',
          source: {
            type: 'base64',
            data: data.data,
            media_type: data.mime_type,
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
        } as AnthropicImageBlockParam;
      } else {
        return {
          type: 'image',
          source: {
            type: 'url',
            url: block.url,
            media_type: block.mime_type ?? '',
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
        } as AnthropicImageBlockParam;
      }
    } else {
      if (block.source_type === 'base64') {
        return {
          type: 'image',
          source: {
            type: 'base64',
            data: block.data,
            media_type: block.mime_type ?? '',
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
        } as AnthropicImageBlockParam;
      } else {
        throw new Error(`Unsupported image source type: ${block.source_type}`);
      }
    }
  },

  fromStandardFileBlock(block: StandardFileBlock): AnthropicDocumentBlockParam {
    const mime_type = (block.mime_type ?? '').split(';')[0];

    if (block.source_type === 'url') {
      if (mime_type === 'application/pdf' || mime_type === '') {
        return {
          type: 'document',
          source: {
            type: 'url',
            url: block.url,
            media_type: block.mime_type ?? '',
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
          ...('citations' in (block.metadata ?? {})
            ? { citations: block.metadata!.citations }
            : {}),
          ...('context' in (block.metadata ?? {})
            ? { context: block.metadata!.context }
            : {}),
          ...('title' in (block.metadata ?? {})
            ? { title: block.metadata!.title }
            : {}),
        } as AnthropicDocumentBlockParam;
      }
      throw new Error(
        `Unsupported file mime type for file url source: ${block.mime_type}`
      );
    } else if (block.source_type === 'text') {
      if (mime_type === 'text/plain' || mime_type === '') {
        return {
          type: 'document',
          source: {
            type: 'text',
            data: block.text,
            media_type: block.mime_type ?? '',
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
          ...('citations' in (block.metadata ?? {})
            ? { citations: block.metadata!.citations }
            : {}),
          ...('context' in (block.metadata ?? {})
            ? { context: block.metadata!.context }
            : {}),
          ...('title' in (block.metadata ?? {})
            ? { title: block.metadata!.title }
            : {}),
        } as AnthropicDocumentBlockParam;
      } else {
        throw new Error(
          `Unsupported file mime type for file text source: ${block.mime_type}`
        );
      }
    } else if (block.source_type === 'base64') {
      if (mime_type === 'application/pdf' || mime_type === '') {
        return {
          type: 'document',
          source: {
            type: 'base64',
            data: block.data,
            media_type: 'application/pdf',
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
          ...('citations' in (block.metadata ?? {})
            ? { citations: block.metadata!.citations }
            : {}),
          ...('context' in (block.metadata ?? {})
            ? { context: block.metadata!.context }
            : {}),
          ...('title' in (block.metadata ?? {})
            ? { title: block.metadata!.title }
            : {}),
        } as AnthropicDocumentBlockParam;
      } else if (
        ['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(
          mime_type
        )
      ) {
        return {
          type: 'document',
          source: {
            type: 'content',
            content: [
              {
                type: 'image',
                source: {
                  type: 'base64',
                  data: block.data,
                  media_type: mime_type as
                    | 'image/jpeg'
                    | 'image/png'
                    | 'image/gif'
                    | 'image/webp',
                },
              },
            ],
          },
          ...('cache_control' in (block.metadata ?? {})
            ? { cache_control: block.metadata!.cache_control }
            : {}),
          ...('citations' in (block.metadata ?? {})
            ? { citations: block.metadata!.citations }
            : {}),
          ...('context' in (block.metadata ?? {})
            ? { context: block.metadata!.context }
            : {}),
          ...('title' in (block.metadata ?? {})
            ? { title: block.metadata!.title }
            : {}),
        } as AnthropicDocumentBlockParam;
      } else {
        throw new Error(
          `Unsupported file mime type for file base64 source: ${block.mime_type}`
        );
      }
    } else {
      throw new Error(`Unsupported file source type: ${block.source_type}`);
    }
  },
};

function _formatContent(message: BaseMessage) {
  const toolTypes = [
    'tool_use',
    'tool_result',
    'input_json_delta',
    'server_tool_use',
    'web_search_tool_result',
    'web_search_result',
  ];
  const textTypes = ['text', 'text_delta'];
  /**
   * Reasoning blocks emitted by other providers — Bedrock's `reasoning_content`,
   * Google's `reasoning`, and LibreChat's `think`. Their signatures are
   * provider-specific and cannot be validated by Anthropic, so on a
   * cross-provider handoff (e.g. Bedrock → Anthropic) we drop them rather than
   * forwarding an unusable block. The receiving model produces its own thinking.
   */
  /**
   * Google server-side tool blocks (`toolCall`/`toolResponse` parts from e.g.
   * URL context or Google Search). Only Google can execute these and validate
   * their thought signatures, so they are dropped on a cross-provider handoff
   * (e.g. Google → Anthropic); the assistant's answer text is kept.
   */
  const foreignServerToolTypes = ['toolCall', 'toolResponse'];
  const { content } = message;

  if (typeof content === 'string') {
    return content;
  } else {
    const contentParts = content as MessageContentComplex[];
    const contentBlocks = contentParts.map((contentPart) => {
      /**
       * Normalize server_tool_use blocks into a clean shape the API accepts.
       * These blocks may arrive with the correct type (server_tool_use), as a
       * standardized server_tool_call, or mislabeled after state serialization.
       * Regardless of current type, if the id starts with 'srvtoolu_' we rebuild
       * a clean block with only the properties the API expects.
       */
      if (
        'id' in contentPart &&
        typeof (contentPart as Record<string, unknown>).id === 'string' &&
        ((contentPart as Record<string, unknown>).id as string).startsWith(
          Constants.ANTHROPIC_SERVER_TOOL_PREFIX
        ) &&
        'name' in contentPart
      ) {
        const rawPart = contentPart as Record<string, unknown>;
        const corrected: AnthropicServerToolUseBlockParam = {
          type: 'server_tool_use',
          id: rawPart.id as string,
          name: (rawPart.name ?? 'web_search') as 'web_search',
          input: coerceAnthropicToolUseInput(rawPart.input ?? rawPart.args),
        };
        return corrected;
      }

      /**
       * Normalize web_search_tool_result blocks into a clean shape.
       * Same rationale as above — the block may carry extra properties from
       * streaming (input, index, etc.) that the API rejects. Rebuild cleanly.
       */
      if (
        'tool_use_id' in contentPart &&
        typeof (contentPart as Record<string, unknown>).tool_use_id ===
          'string' &&
        (
          (contentPart as Record<string, unknown>).tool_use_id as string
        ).startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) &&
        'content' in contentPart
      ) {
        const rawPart = contentPart as Record<string, unknown>;
        const content = rawPart.content;
        const isValidContent =
          Array.isArray(content) ||
          (content != null &&
            typeof content === 'object' &&
            'type' in content &&
            (content as Record<string, unknown>).type ===
              'web_search_tool_result_error');

        if (isValidContent) {
          const corrected: AnthropicWebSearchToolResultBlockParam = {
            type: 'web_search_tool_result',
            tool_use_id: rawPart.tool_use_id as string,
            content:
              content as AnthropicWebSearchToolResultBlockParam['content'],
          };
          return corrected;
        }
        return null;
      }

      /**
       * Native ChatModelStream output standardizes client tool uses as
       * `tool_call` blocks. Convert those blocks back to Anthropic's wire shape
       * before the generic content formatter sees them.
       */
      if (
        contentPart.type === 'tool_call' &&
        'id' in contentPart &&
        typeof contentPart.id === 'string' &&
        'name' in contentPart &&
        typeof contentPart.name === 'string' &&
        'args' in contentPart
      ) {
        let args = contentPart.args;
        if (typeof args === 'string') {
          try {
            args = JSON.parse(args);
          } catch {
            args = {};
          }
        }
        if (args == null || typeof args !== 'object' || Array.isArray(args)) {
          args = {};
        }
        return _convertLangChainToolCallToAnthropic({
          id: contentPart.id,
          name: contentPart.name,
          args: args as Record<string, unknown>,
        });
      }

      /**
       * Native Anthropic thinking is exposed through the standard reasoning
       * stream, but its signature makes it safe and necessary to round-trip.
       * Keep unsigned or cross-provider reasoning on the existing drop path.
       */
      if (
        contentPart.type === 'reasoning' &&
        isAIMessage(message) &&
        message.response_metadata.model_provider === 'anthropic' &&
        'reasoning' in contentPart &&
        typeof contentPart.reasoning === 'string' &&
        'signature' in contentPart &&
        typeof contentPart.signature === 'string' &&
        contentPart.signature !== ''
      ) {
        const block: AnthropicThinkingBlockParam = {
          type: 'thinking',
          thinking: contentPart.reasoning,
          signature: contentPart.signature,
        };
        return block;
      }

      /**
       * Skip non-server malformed blocks that have tool fields mixed with text type.
       */
      if (
        'id' in contentPart &&
        'name' in contentPart &&
        'input' in contentPart &&
        contentPart.type === 'text'
      ) {
        return null;
      }
      if (
        'tool_use_id' in contentPart &&
        'content' in contentPart &&
        contentPart.type === 'text'
      ) {
        return null;
      }

      // Core's v1 streaming aggregation can leave a partial tool-input delta as a
      // standalone block typed `text` carrying `input` but no `text`. The assembled
      // input is restored on the tool_use block from `message.tool_calls`, so drop it.
      if (
        contentPart.type === 'text' &&
        'input' in contentPart &&
        !('text' in contentPart)
      ) {
        return null;
      }

      if (isDataContentBlock(contentPart)) {
        return convertToProviderContentBlock(
          contentPart,
          standardContentBlockConverter
        );
      }

      const cacheControl =
        'cache_control' in contentPart ? contentPart.cache_control : undefined;

      if (contentPart.type === 'image_url') {
        let source;
        const imageUrl = (contentPart as ImageUrlContentBlock).image_url;
        if (typeof imageUrl === 'string') {
          source = _formatImage(imageUrl);
        } else {
          source = _formatImage(imageUrl.url);
        }
        return {
          type: 'image' as const, // Explicitly setting the type as "image"
          source,
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
        };
      } else if (isAnthropicImageBlockParam(contentPart)) {
        return contentPart;
      } else if (contentPart.type === 'document') {
        // PDF
        return {
          ...contentPart,
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
        };
      } else if (contentPart.type === 'thinking') {
        const thinkingPart = contentPart as AnthropicThinkingBlockParam;
        // Google thinking-enabled output reuses `type: 'thinking'` but carries
        // no Anthropic signature. Anthropic rejects an unsigned thinking block,
        // so on an assistant turn treat it as foreign reasoning and drop it
        // rather than forward an unusable block. Signed (Anthropic-native)
        // thinking is forwarded as before.
        const signature = (thinkingPart as { signature?: string }).signature;
        if (isAIMessage(message) && (signature == null || signature === '')) {
          return null;
        }
        // Opus 4.7+ omits thinking text by default (signed but text-less
        // block), so `thinking` may be absent on a signed block even though the
        // type declares it required. Coerce to '' — JSON.stringify drops an
        // undefined value, which would send a `thinking` block missing its
        // required `thinking` field and trip a 400
        // `messages.N.content.M.thinking.thinking: Field required`.
        const thinkingText =
          (thinkingPart as { thinking?: string }).thinking ?? '';
        const block: AnthropicThinkingBlockParam = {
          type: 'thinking' as const, // Explicitly setting the type as "thinking"
          thinking: thinkingText,
          signature: thinkingPart.signature,
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
        };
        return block;
      } else if (contentPart.type === 'redacted_thinking') {
        const redactedPart = contentPart as AnthropicRedactedThinkingBlockParam;
        const block: AnthropicRedactedThinkingBlockParam = {
          type: 'redacted_thinking' as const, // Explicitly setting the type as "redacted_thinking"
          data: redactedPart.data,
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
        };
        return block;
      } else if (contentPart.type === 'search_result') {
        const searchResultPart = contentPart as AnthropicSearchResultBlockParam;
        const block: AnthropicSearchResultBlockParam = {
          type: 'search_result' as const,
          title: searchResultPart.title,
          source: searchResultPart.source,
          ...('cache_control' in contentPart &&
          contentPart.cache_control != null
            ? { cache_control: contentPart.cache_control }
            : {}),
          ...('citations' in contentPart && contentPart.citations != null
            ? { citations: contentPart.citations }
            : {}),
          content: searchResultPart.content,
        };
        return block;
      } else if (contentPart.type === 'compaction') {
        const compactionPart = contentPart as AnthropicCompactionBlockParam;
        const block: AnthropicCompactionBlockParam = {
          type: 'compaction' as const,
          content: compactionPart.content,
          ...('encrypted_content' in compactionPart
            ? { encrypted_content: compactionPart.encrypted_content }
            : {}),
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
        };
        return block;
      } else if (
        textTypes.some((t) => t === contentPart.type) &&
        'text' in contentPart
      ) {
        // Assuming contentPart is of type MessageContentText here
        return {
          type: 'text' as const, // Explicitly setting the type as "text"
          text: contentPart.text,
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
          ...('citations' in contentPart && contentPart.citations != null
            ? { citations: contentPart.citations }
            : {}),
        };
      } else if (toolTypes.some((t) => t === contentPart.type)) {
        const contentPartCopy = { ...contentPart };
        if ('index' in contentPartCopy) {
          // Anthropic does not support passing the index field here, so we remove it.
          delete contentPartCopy.index;
        }

        if (contentPartCopy.type === 'input_json_delta') {
          // Orphaned partial tool-input delta with no id of its own. The assembled
          // input is restored on the tool_use block from `message.tool_calls`; drop it.
          return null;
        }

        if (
          contentPartCopy.type === 'tool_use' &&
          'id' in contentPartCopy &&
          typeof contentPartCopy.id === 'string' &&
          contentPartCopy.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
        ) {
          contentPartCopy.type = 'server_tool_use';
        }

        // Core's streaming aggregation can leave the inline tool_use input empty
        // (the assembled arguments live in `message.tool_calls` or, for persisted
        // messages, in sibling input_json_delta blocks). Restore it when missing.
        // An empty plain object counts as missing too: context-pressure
        // truncation degrades an inline input to `{}` while a small `args`
        // mirror survives intact, and a genuinely empty call's mirror is also
        // `{}`, so preferring the mirror never loses information.
        if (
          contentPartCopy.type === 'tool_use' &&
          typeof contentPartCopy.id === 'string' &&
          (contentPartCopy.input === '' ||
            contentPartCopy.input == null ||
            isEmptyPlainObject(contentPartCopy.input))
        ) {
          const matchingToolCall = isAIMessage(message)
            ? message.tool_calls?.find(
              (toolCall) => toolCall.id === contentPartCopy.id
            )
            : undefined;
          if (matchingToolCall) {
            contentPartCopy.input = matchingToolCall.args;
          } else {
            const blockIndex = (contentPart as Record<string, unknown>).index;
            const merged = contentParts
              .filter((part) => {
                const p = part as Record<string, unknown>;
                return (
                  p.type === 'input_json_delta' &&
                  p.index === blockIndex &&
                  typeof p.input === 'string'
                );
              })
              .reduce(
                (acc, part) => acc + (part as Record<string, unknown>).input,
                ''
              );
            if (merged !== '') {
              contentPartCopy.input = merged;
            }
          }
        }

        if (
          contentPartCopy.type === 'tool_use' ||
          contentPartCopy.type === 'server_tool_use'
        ) {
          /**
           * The API requires `input` to be a JSON object. Streaming leaves the
           * raw partial-JSON string here, and context-pressure truncation can
           * leave `null` on BOTH the block and its `tool_calls` mirror — so
           * the restore above may itself restore a non-object. Coerce last,
           * after every restore path has run.
           */
          contentPartCopy.input = coerceAnthropicToolUseInput(
            'input' in contentPartCopy ? contentPartCopy.input : undefined
          );
        } else if (
          'input' in contentPartCopy &&
          typeof contentPartCopy.input === 'string'
        ) {
          // Non-tool_use tool blocks keep their legacy string handling.
          try {
            contentPartCopy.input = JSON.parse(contentPartCopy.input);
          } catch {
            contentPartCopy.input = {};
          }
        }

        /**
         * For multi-turn conversations with citations, we must preserve ALL blocks
         * including server_tool_use, web_search_tool_result, and web_search_result.
         * Citations reference search results by index, so filtering changes indices and breaks references.
         *
         * The ToolNode already handles skipping server tool invocations via the srvtoolu_ prefix check.
         */

        // TODO: Fix when SDK types are fixed
        return {
          ...contentPartCopy,
          ...(cacheControl != null ? { cache_control: cacheControl } : {}),
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      } else if (
        'functionCall' in contentPart &&
        contentPart.functionCall != null &&
        typeof contentPart.functionCall === 'object' &&
        isAIMessage(message)
      ) {
        const functionCallPart = contentPart as GoogleFunctionCallBlock;
        const correspondingToolCall = message.tool_calls?.find(
          (toolCall) => toolCall.name === functionCallPart.functionCall.name
        );
        if (!correspondingToolCall) {
          throw new Error(
            `Could not find tool call for function call ${functionCallPart.functionCall.name}`
          );
        }
        // Google GenAI models include a `functionCall` object inside content. We should ignore it as Anthropic will not support it.
        return {
          id: correspondingToolCall.id,
          type: 'tool_use',
          name: correspondingToolCall.name,
          input: functionCallPart.functionCall.args,
        };
      } else if (isAIMessage(message) && isReasoningContentBlock(contentPart)) {
        // Foreign reasoning on an ASSISTANT turn (Bedrock `reasoning_content`,
        // Google `reasoning`, LibreChat `think`) carries provider-specific
        // signatures Anthropic cannot validate; drop it so a cross-provider
        // handoff doesn't crash. The same types on a user/tool turn are real
        // input and fall through to the throw below rather than being silently
        // dropped — as does any other unknown block (user media, Google
        // code-execution), which must be surfaced, not discarded.
        return null;
      } else if (
        isAIMessage(message) &&
        foreignServerToolTypes.some((t) => t === contentPart.type)
      ) {
        // Google server-side tool call/response (e.g. URL context) — only
        // Google can execute it or validate its thought signature; drop it
        // on a cross-provider handoff rather than crash.
        return null;
      } else {
        console.error(
          'Unsupported content part:',
          JSON.stringify(contentPart, null, 2)
        );
        throw new Error('Unsupported message content format');
      }
    });
    const filteredContentBlocks = contentBlocks.filter(
      (block) =>
        block !== null &&
        !(
          block.type === 'text' &&
          'text' in block &&
          typeof block.text === 'string' &&
          block.text.trim() === ''
        )
    );
    return filteredContentBlocks.length > 0
      ? filteredContentBlocks
      : [{ type: 'text' as const, text: ANTHROPIC_EMPTY_TEXT_PLACEHOLDER }];
  }
}

/**
 * Formats messages as a prompt for the model.
 * Used in LangSmith, export is important here.
 * @param messages The base messages to format as a prompt.
 * @returns The formatted prompt.
 */
export function _convertMessagesToAnthropicPayload(
  messages: BaseMessage[]
): AnthropicMessageCreateParams {
  const mergedMessages = _ensureMessageContents(messages);
  let system;
  if (mergedMessages.length > 0 && mergedMessages[0]._getType() === 'system') {
    system = messages[0].content;
  }
  const conversationMessages =
    system !== undefined ? mergedMessages.slice(1) : mergedMessages;
  const formattedMessages = conversationMessages.map((message) => {
    let role;
    if (message._getType() === 'human') {
      role = 'user' as const;
    } else if (message._getType() === 'ai') {
      role = 'assistant' as const;
    } else if (message._getType() === 'tool') {
      role = 'user' as const;
    } else if (message._getType() === 'system') {
      throw new Error(
        'System messages are only permitted as the first passed message.'
      );
    } else {
      throw new Error(`Message type "${message._getType()}" is not supported.`);
    }
    const isAI = isAIMessage(message);
    const toolCalls = isAI ? (message.tool_calls ?? []) : [];
    if (isAI && toolCalls.length > 0) {
      if (typeof message.content === 'string') {
        const clientToolCalls = toolCalls.filter(
          (tc) =>
            !(
              tc.id?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) ?? false
            )
        );
        if (message.content === '') {
          return {
            role,
            content:
              clientToolCalls.length > 0
                ? clientToolCalls.map(_convertLangChainToolCallToAnthropic)
                : [
                  {
                    type: 'text' as const,
                    text: ANTHROPIC_EMPTY_TEXT_PLACEHOLDER,
                  },
                ],
          };
        } else {
          return {
            role,
            content: [
              { type: 'text' as const, text: message.content },
              ...clientToolCalls.map(_convertLangChainToolCallToAnthropic),
            ],
          };
        }
      } else {
        const formattedContent = _formatContent(message);
        const formattedBlocks = Array.isArray(formattedContent)
          ? formattedContent
          : [];
        // Tool calls already materialized as content blocks by `_formatContent`.
        // Derived from the FORMATTED output (not the raw content by type) so
        // that Google `functionCall` parts — which `_formatContent` converts
        // into `tool_use` — count as represented and are not appended twice.
        const representedToolIds = new Set(
          formattedBlocks
            .filter(
              (block) =>
                block != null &&
                (block.type === 'tool_use' || block.type === 'server_tool_use')
            )
            .map((block) => (block as { id?: string }).id)
        );
        // Client tool calls present in `tool_calls` but absent from the
        // formatted content — e.g. a Bedrock extended-thinking turn records the
        // tool only on `tool_calls` and leaves `content` as just the reasoning
        // block. Without materializing them, dropping that reasoning block
        // silently loses the (handoff) tool call instead of forwarding it.
        const unrepresentedToolCalls = toolCalls.filter(
          (toolCall) =>
            !(
              toolCall.id?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) ??
              false
            ) && !representedToolIds.has(toolCall.id)
        );
        if (unrepresentedToolCalls.length === 0) {
          return { role, content: formattedContent };
        }
        const existingBlocks = formattedBlocks.filter(
          (block) =>
            !(
              block != null &&
              block.type === 'text' &&
              'text' in block &&
              block.text === ANTHROPIC_EMPTY_TEXT_PLACEHOLDER
            )
        );
        return {
          role,
          content: [
            ...existingBlocks,
            ...unrepresentedToolCalls.map(_convertLangChainToolCallToAnthropic),
          ],
        };
      }
    } else {
      return {
        role,
        content: _formatContent(message),
      };
    }
  });
  return {
    messages: mergeMessages(formattedMessages),
    system,
  } as AnthropicMessageCreateParams;
}

/** Claude 4.6 and every later release (5.x, Fable, Mythos) reject a trailing assistant prefill. */
export function modelDisallowsAssistantPrefill(model?: string): boolean {
  const match = CLAUDE_FAMILY_VERSION_PATTERN.exec(model ?? '');
  if (!match) {
    return false;
  }
  const [, majorText, minorText = '0'] = match;
  const major = Number(majorText);
  const minor = Number(minorText);
  return major > 4 || (major === 4 && minor >= 6);
}

function messagesHaveCacheControl(
  messages: AnthropicMessageCreateParams['messages']
): boolean {
  return messages.some(
    (message) =>
      Array.isArray(message.content) &&
      message.content.some((block) => 'cache_control' in block)
  );
}

/**
 * Place one ephemeral `cache_control` on the last cacheable block of the final
 * message of an already-converted Anthropic payload. Used to re-anchor the tail
 * breakpoint after a trailing assistant prefill is stripped. Operates on the
 * post-conversion payload, where blocks the converter drops (foreign reasoning,
 * input_json_delta) are already gone — only native thinking blocks must be
 * skipped. Returns a new array only when it actually places a marker.
 */
function reanchorTailCacheControl(
  messages: AnthropicMessageCreateParams['messages'],
  ttl?: '1h'
): AnthropicMessageCreateParams['messages'] {
  if (messages.length === 0) {
    return messages;
  }
  const cacheControl =
    ttl === '1h'
      ? ({ type: 'ephemeral', ttl: '1h' } as const)
      : ({ type: 'ephemeral' } as const);
  const lastIndex = messages.length - 1;
  const tail = messages[lastIndex];
  const content = tail.content;

  if (typeof content === 'string') {
    if (content.trim() === '') {
      return messages;
    }
    const next = [...messages];
    next[lastIndex] = {
      ...tail,
      content: [{ type: 'text', text: content, cache_control: cacheControl }],
    } as (typeof messages)[number];
    return next;
  }

  if (!Array.isArray(content)) {
    return messages;
  }

  let anchor = -1;
  for (let i = 0; i < content.length; i++) {
    const block = content[i] as { type?: string };
    const type = block.type;
    if (type == null || isReasoningContentBlock(block)) {
      continue;
    }
    if (
      type === 'text' &&
      ((content[i] as { text?: string }).text ?? '').trim() === ''
    ) {
      continue;
    }
    anchor = i;
  }
  if (anchor < 0) {
    return messages;
  }

  const next = [...messages];
  next[lastIndex] = {
    ...tail,
    content: content.map((block, i) =>
      i === anchor ? { ...block, cache_control: cacheControl } : block
    ),
  } as (typeof messages)[number];
  return next;
}

/**
 * Find the extended-cache TTL (`'1h'`) carried by an existing `cache_control`
 * breakpoint, so {@link reanchorTailCacheControl} can re-apply the same TTL the
 * stripped prefill had. Returns `undefined` for the legacy 5-minute default
 * (no `ttl`), keeping that path byte-identical to before.
 */
function findCacheControlTtl(
  messages: AnthropicMessageCreateParams['messages']
): '1h' | undefined {
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      const cacheControl = (block as { cache_control?: { ttl?: unknown } })
        .cache_control;
      if (cacheControl?.ttl === '1h') {
        return '1h';
      }
    }
  }
  return undefined;
}

export function stripUnsupportedAssistantPrefill<
  T extends Pick<AnthropicMessageCreateParams, 'messages'> & { model?: string },
>(request: T): T {
  if (!modelDisallowsAssistantPrefill(request.model)) {
    return request;
  }

  const messages = request.messages;
  if (
    messages.length <= 1 ||
    messages[messages.length - 1]?.role !== 'assistant'
  ) {
    return request;
  }

  const nextMessages = [...messages];
  while (
    nextMessages.length > 1 &&
    nextMessages[nextMessages.length - 1]?.role === 'assistant'
  ) {
    nextMessages.pop();
  }

  /**
   * If a single tail prompt-cache breakpoint rode the stripped assistant
   * prefill, the survivors may now carry no `cache_control` at all, dropping
   * message caching for this request. Re-anchor the breakpoint on the new tail
   * (only when one was actually lost, so caching-off requests stay untouched).
   */
  const reanchored =
    messagesHaveCacheControl(messages) &&
    !messagesHaveCacheControl(nextMessages)
      ? reanchorTailCacheControl(nextMessages, findCacheControlTtl(messages))
      : nextMessages;

  return {
    ...request,
    messages: reanchored,
  };
}

function mergeMessages(messages: AnthropicMessageCreateParams['messages']) {
  if (messages.length <= 1) {
    return messages;
  }

  const result: AnthropicMessageCreateParams['messages'] = [];
  let currentMessage = messages[0];

  type ContentBlocks = Exclude<
    AnthropicMessageCreateParams['messages'][number]['content'],
    string
  >;
  const normalizeContent = (
    content: AnthropicMessageCreateParams['messages'][number]['content']
  ): ContentBlocks => {
    if (typeof content === 'string') {
      return [{ type: 'text', text: content }];
    }
    return content;
  };

  const isToolResultMessage = (msg: (typeof messages)[0]) => {
    if (msg.role !== 'user') return false;

    if (typeof msg.content === 'string') {
      return false;
    }

    return (
      Array.isArray(msg.content) &&
      msg.content.every((item) => item.type === 'tool_result')
    );
  };

  for (let i = 1; i < messages.length; i += 1) {
    const nextMessage = messages[i];

    if (
      isToolResultMessage(currentMessage) &&
      isToolResultMessage(nextMessage)
    ) {
      // Merge the messages by combining their content arrays
      currentMessage = {
        ...currentMessage,
        content: [
          ...normalizeContent(currentMessage.content),
          ...normalizeContent(nextMessage.content),
        ],
      };
    } else {
      result.push(currentMessage);
      currentMessage = nextMessage;
    }
  }

  result.push(currentMessage);
  return result;
}

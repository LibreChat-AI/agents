import { isProxy } from 'node:util/types';
import { BaseMessage, MessageContentComplex } from '@langchain/core/messages';
import type Anthropic from '@anthropic-ai/sdk';
import type { AnthropicMessage } from '@/types/messages';
import { hasComputerCallOutputMarker } from '@/utils/toolContent';
import { toLangChainContent } from './langchain';
import { isReasoningContentBlock } from './reasoningTypes';
import { ContentTypes } from '@/common/enum';

type MessageWithContent = {
  content?: string | MessageContentComplex[];
};

type MessageContentWithCacheControl = MessageContentComplex & {
  cache_control?: unknown;
};

/**
 * Prompt-cache breakpoint TTL.
 *
 * Both Anthropic (`cache_control.ttl`) and Bedrock (`cachePoint.ttl`) accept
 * `'5m'` (the legacy provider default) and `'1h'` (the extended cache). When
 * prompt caching is enabled the SDK now defaults to the 1-hour extended cache
 * (see {@link DEFAULT_PROMPT_CACHE_TTL}); pass `'5m'` to opt back into the
 * legacy 5-minute behavior.
 */
export type PromptCacheTtl = '5m' | '1h';

/**
 * Default TTL applied wherever a prompt-cache breakpoint is added. The 1-hour
 * extended cache keeps prefixes warm across longer gaps between turns, at the
 * cost of a higher one-time cache-write multiplier (2x vs 1.25x for 5m).
 */
export const DEFAULT_PROMPT_CACHE_TTL: PromptCacheTtl = '1h';

/**
 * Resolve an optionally-configured TTL to a concrete value, defaulting to the
 * 1-hour extended cache. Used at the Anthropic/Bedrock prompt-cache call sites.
 */
export function resolvePromptCacheTtl(
  ttl: PromptCacheTtl | undefined
): PromptCacheTtl {
  return ttl ?? DEFAULT_PROMPT_CACHE_TTL;
}

/** Anthropic `cache_control` shape (the SDK accepts an optional `ttl`). */
type AnthropicCacheControl = { type: 'ephemeral'; ttl?: '1h' };

/**
 * Build an Anthropic `cache_control` breakpoint for the given TTL. `'5m'` (or
 * `undefined`) omits the `ttl` field — that is the provider default, so the
 * payload stays byte-identical to the legacy 5-minute marker. `'1h'` adds the
 * explicit extended-cache `ttl`.
 */
export function buildAnthropicCacheControl(
  ttl?: PromptCacheTtl
): AnthropicCacheControl {
  return ttl === '1h'
    ? { type: 'ephemeral', ttl: '1h' }
    : { type: 'ephemeral' };
}

/** Bedrock `cachePoint` shape (the SDK accepts an optional `ttl`). */
type BedrockCachePoint = { type: 'default'; ttl?: '1h' };

/**
 * Build a Bedrock `cachePoint` for the given TTL. Mirrors
 * {@link buildAnthropicCacheControl}: `'5m'`/`undefined` omits `ttl` (the
 * legacy default), `'1h'` adds the extended-cache `ttl`.
 */
export function buildBedrockCachePoint(
  ttl?: PromptCacheTtl
): BedrockCachePoint {
  return ttl === '1h' ? { type: 'default', ttl: '1h' } : { type: 'default' };
}

/**
 * Whether a Bedrock model id is an Anthropic Claude model. Claude is the only
 * Bedrock family that accepts the explicit cache features below; Amazon Nova and
 * others reject them. Matches both the `anthropic.` and cross-region
 * (`us.anthropic.`) / bare `claude` forms.
 */
function isBedrockAnthropicModel(model: string | undefined | null): boolean {
  return typeof model === 'string' && /claude|anthropic/i.test(model);
}

/**
 * A `cachePoint` under `toolConfig.tools` is only accepted on Anthropic Claude
 * models. Amazon Nova rejects it outright — `Malformed input request:
 * #/toolConfig/tools/0: extraneous key [cachePoint] is not permitted` — even
 * though it accepts `cachePoint` in `system` and `messages` (verified live:
 * Nova returns HTTP 200 with real `cacheWriteInputTokens` for both). Per the AWS
 * prompt-caching docs the support table lists `tools` for Claude only.
 *
 * Gate ONLY the tool checkpoint on this, so a `promptCache: true` config on Nova
 * keeps its valid message/system caching instead of either 400-ing (tool point)
 * or losing caching entirely.
 *
 * @see https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
 * @see https://github.com/danny-avila/LibreChat/issues/13838
 */
export function supportsBedrockToolCache(
  model: string | undefined | null
): boolean {
  return isBedrockAnthropicModel(model);
}

/**
 * Resolve the Bedrock `cachePoint` TTL for a model. The extended `1h` TTL is
 * Anthropic-only: non-Claude models (e.g. Nova) hard-reject it with
 * `Extended TTL prompt caching is only supported for Anthropic models`, so they
 * are clamped to `5m` even when `1h` is configured (verified live). An omitted
 * model defaults to a Claude model, and Claude downgrades an unsupported `1h` to
 * `5m` server-side, so the `1h` default is safe for the Claude/undefined paths.
 *
 * @see https://github.com/danny-avila/LibreChat/issues/13838
 */
export function resolveBedrockPromptCacheTtl(
  ttl: PromptCacheTtl | undefined,
  model: string | undefined | null
): PromptCacheTtl {
  if (model != null && !isBedrockAnthropicModel(model)) {
    return '5m';
  }
  return resolvePromptCacheTtl(ttl);
}

/**
 * Deep clones a message's content to prevent mutation of the original.
 */
function deepCloneContent<T extends string | MessageContentComplex[]>(
  content: T,
  omittedProperty?: string
): T {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((block) => {
      if (typeof block !== 'object' || block == null || isProxy(block)) {
        return block;
      }
      try {
        const descriptors = Object.getOwnPropertyDescriptors(block);
        if (omittedProperty != null) {
          delete descriptors[omittedProperty];
        }
        return Object.create(
          Object.getPrototypeOf(block),
          descriptors
        ) as MessageContentComplex;
      } catch {
        return block;
      }
    }) as T;
  }
  return content;
}

/**
 * Clones a message with new content. For LangChain BaseMessage instances,
 * constructs a proper class instance so that `instanceof` checks are preserved
 * in downstream code (e.g., ensureThinkingBlockInMessages).
 * For plain objects (AnthropicMessage), uses object spread.
 */
export function cloneMessage<T extends MessageWithContent>(
  message: T,
  content: string | MessageContentComplex[]
): T {
  if (message instanceof BaseMessage) {
    const descriptors = Object.getOwnPropertyDescriptors(message) as Record<
      PropertyKey,
      PropertyDescriptor | undefined
    >;
    const contentDescriptor = descriptors.content;
    descriptors.content = {
      configurable: contentDescriptor?.configurable ?? true,
      enumerable: contentDescriptor?.enumerable ?? true,
      value: toLangChainContent(content),
      writable: contentDescriptor?.writable ?? true,
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
          content: toLangChainContent(content),
        },
      };
    }
    return Object.create(
      Object.getPrototypeOf(message),
      descriptors as PropertyDescriptorMap
    ) as T;
  }

  const {
    lc_kwargs: _lc_kwargs,
    lc_serializable: _lc_serializable,
    lc_namespace: _lc_namespace,
    ...rest
  } = message as T & {
    lc_kwargs?: unknown;
    lc_serializable?: unknown;
    lc_namespace?: unknown;
  };

  const cloned = { ...rest, content } as T;

  // LangChain messages don't have a direct 'role' property - derive it from getType()
  if (
    'getType' in message &&
    typeof message.getType === 'function' &&
    !('role' in cloned)
  ) {
    const msgType = (message as unknown as BaseMessage).getType();
    const roleMap: Record<string, string> = {
      human: 'user',
      ai: 'assistant',
      system: 'system',
      tool: 'tool',
    };
    (cloned as Record<string, unknown>).role = roleMap[msgType] || msgType;
  }

  return cloned;
}

/**
 * Sanitize a Bedrock system message: strip Anthropic `cache_control` (Bedrock
 * conversion can't use it) and normalize any existing `cachePoint` to the
 * resolved TTL. The normalization matters because Bedrock requires longer-TTL
 * checkpoints to appear before shorter ones — a stale 5-minute system cachePoint
 * (host-supplied or carried over from a 5m config) left ahead of a 1-hour
 * message tail would make the request invalid. System messages are never
 * anchored as the tail breakpoint; this only fixes markers already present.
 */
function sanitizeBedrockSystemMessage<T extends MessageWithContent>(
  message: T,
  ttl?: PromptCacheTtl
): T {
  const content = message.content;
  if (!Array.isArray(content)) {
    return message;
  }

  const sanitized: MessageContentComplex[] = [];
  let modified = false;
  for (const block of content) {
    if (isCachePoint(block)) {
      const descriptor = Object.getOwnPropertyDescriptor(block, 'cachePoint');
      const existing =
        descriptor != null && 'value' in descriptor
          ? (descriptor.value as { ttl?: unknown } | undefined)
          : undefined;
      const desired = buildBedrockCachePoint(ttl);
      if (existing?.ttl !== desired.ttl) {
        modified = true;
        sanitized.push({ cachePoint: desired } as MessageContentComplex);
      } else {
        sanitized.push(block);
      }
      continue;
    }
    if ('cache_control' in block) {
      const cloned: MessageContentWithCacheControl = { ...block };
      delete cloned.cache_control;
      modified = true;
      sanitized.push(cloned);
      continue;
    }
    sanitized.push(block);
  }

  if (!modified) {
    return message;
  }

  return cloneMessage(message, sanitized);
}

/**
 * Anthropic API: Adds cache control to the appropriate user messages in the payload.
 * Strips ALL existing cache control (both Anthropic and Bedrock formats) from all messages,
 * then adds fresh cache control to the last 2 user messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * Returns a new array - only clones messages that require modification.
 * @param messages - The array of message objects.
 * @returns - A new array of message objects with cache control added.
 */
export function addCacheControl<T extends AnthropicMessage | BaseMessage>(
  messages: T[],
  ttl?: PromptCacheTtl
): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let userMessagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;
    const isUserMessage =
      ('getType' in originalMessage && originalMessage.getType() === 'human') ||
      ('role' in originalMessage && originalMessage.role === 'user');
    const hasArrayContent = Array.isArray(content);
    const needsCacheAdd =
      userMessagesModified < 2 &&
      isUserMessage &&
      !isSyntheticMetaMessage(originalMessage) &&
      (typeof content === 'string' || hasArrayContent);

    // Skip messages that don't need any work
    if (!needsCacheAdd && !hasArrayContent) {
      continue;
    }

    let workingContent: MessageContentComplex[];
    let modified = false;

    if (hasArrayContent) {
      // Single pass: clone blocks, strip cache markers and cache points,
      // find last text block index for cache insertion — all at once.
      const src = content as MessageContentComplex[];
      workingContent = [];
      let lastTextIndex = -1;
      for (let j = 0; j < src.length; j++) {
        const block = src[j];
        if (isCachePoint(block)) {
          modified = true;
          continue; // skip cache point blocks
        }
        const cloned = { ...block };
        if ('cache_control' in cloned) {
          delete (cloned as Record<string, unknown>).cache_control;
          modified = true;
        }
        if ('type' in cloned && cloned.type === 'text') {
          lastTextIndex = workingContent.length;
        }
        workingContent.push(cloned as MessageContentComplex);
      }

      if (!modified && !needsCacheAdd) {
        continue; // nothing to strip and no cache to add
      }

      // Add cache control to the last text block for user messages
      if (needsCacheAdd && lastTextIndex >= 0) {
        (
          workingContent[lastTextIndex] as Anthropic.TextBlockParam
        ).cache_control = buildAnthropicCacheControl(ttl);
        userMessagesModified++;
      }
    } else if (typeof content === 'string' && needsCacheAdd) {
      workingContent = [
        {
          type: 'text',
          text: content,
          cache_control: buildAnthropicCacheControl(ttl),
        },
      ] as unknown as MessageContentComplex[];
      userMessagesModified++;
    } else {
      continue;
    }

    updatedMessages[i] = cloneMessage(
      originalMessage as MessageWithContent,
      workingContent
    ) as T;
  }

  return updatedMessages;
}

/**
 * Checks if a content block is a cache point
 */
function isCachePoint(block: unknown): boolean {
  if (typeof block !== 'object' || block == null || isProxy(block)) {
    return false;
  }
  try {
    const cachePoint = Object.getOwnPropertyDescriptor(block, 'cachePoint');
    const type = Object.getOwnPropertyDescriptor(block, 'type');
    return cachePoint?.enumerable === true && type == null;
  } catch {
    return false;
  }
}

/**
 * Block types that must never anchor the tail cache breakpoint, because the
 * marker would not survive to the model call:
 * - `thinking` / `redacted_thinking`: native Anthropic reasoning — the API
 *   rejects `cache_control` on these blocks.
 * - `reasoning_content` / `reasoning` / `think`: foreign reasoning (Bedrock,
 *   Google, LibreChat) that `_convertMessagesToAnthropicPayload` DROPS on
 *   assistant turns during a cross-provider handoff.
 * - `input_json_delta`: persisted partial tool-input deltas, also DROPPED by
 *   `_convertMessagesToAnthropicPayload` (the assembled input is restored onto
 *   the tool_use block).
 * Anchoring the only breakpoint on a block that is about to disappear silently
 * loses tail caching, so all of these are excluded.
 */
const NON_ANCHORABLE_BLOCK_TYPES = new Set([
  'input_json_delta',
]);

/**
 * A block can anchor the tail cache breakpoint when it is a real content block
 * that the Anthropic API accepts `cache_control` on and that survives provider
 * conversion. Reasoning and dropped-delta blocks are excluded, and empty text
 * blocks are not cacheable, so all are skipped.
 */
function isTailCacheableBlock(block: MessageContentComplex): boolean {
  if (isCachePoint(block)) {
    return false;
  }
  const type = (block as { type?: string }).type;
  if (
    type == null ||
    isReasoningContentBlock(block) ||
    NON_ANCHORABLE_BLOCK_TYPES.has(type)
  ) {
    return false;
  }
  if (type === 'text') {
    const text = (block as { text?: string }).text;
    return text != null && text.trim() !== '';
  }
  return true;
}

/**
 * Anthropic API: single tail cache breakpoint (default strategy).
 *
 * Places exactly ONE `cache_control` marker on the last cacheable block of the
 * final non-synthetic message, mirroring the Claude Code strategy
 * (`markerIndex = messages.length - 1`). Because the marker always rides the
 * true tail, the entire conversation prefix is written once and read back on
 * the next turn as the history grows append-only — instead of the rolling
 * "last two user messages" markers, which leave freshly appended tool/assistant
 * turns outside the cached prefix and re-write large spans every step.
 *
 * Stale markers (Anthropic `cache_control` and Bedrock cache points) are
 * stripped from every message in a single backward pass so exactly one marker
 * survives. Synthetic skill/meta messages are skipped as anchors (their volatile
 * content must not pin the cache) but still have stale markers removed.
 *
 * Returns a new array; only messages that require modification are cloned.
 */
export function addTailCacheControl<T extends AnthropicMessage | BaseMessage>(
  messages: T[],
  ttl?: PromptCacheTtl
): T[] {
  if (!Array.isArray(messages) || messages.length === 0) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let markerPlaced = false;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;
    const hasArrayContent = Array.isArray(content);
    const isComputerOutput =
      originalMessage instanceof BaseMessage &&
      hasComputerCallOutputMarker(originalMessage);
    const canPlaceMarker =
      !markerPlaced &&
      !isSyntheticMetaMessage(originalMessage) &&
      !isComputerOutput;

    // Earlier string-content messages carry no markers to strip.
    if (!canPlaceMarker && !hasArrayContent) {
      continue;
    }

    let workingContent: MessageContentComplex[];
    let modified = false;

    if (hasArrayContent) {
      const src = content as MessageContentComplex[];
      workingContent = [];
      let tailIndex = -1;
      for (let j = 0; j < src.length; j++) {
        const block = src[j];
        if (isCachePoint(block)) {
          modified = true;
          continue;
        }
        const cloned = { ...block };
        if ('cache_control' in cloned) {
          delete (cloned as Record<string, unknown>).cache_control;
          modified = true;
        }
        if (
          canPlaceMarker &&
          isTailCacheableBlock(cloned as MessageContentComplex)
        ) {
          tailIndex = workingContent.length;
        }
        workingContent.push(cloned as MessageContentComplex);
      }

      if (canPlaceMarker && tailIndex >= 0) {
        (workingContent[tailIndex] as Anthropic.TextBlockParam).cache_control =
          buildAnthropicCacheControl(ttl);
        markerPlaced = true;
        modified = true;
      }

      if (!modified) {
        continue;
      }
    } else if (
      typeof content === 'string' &&
      canPlaceMarker &&
      content.trim() !== ''
    ) {
      workingContent = [
        {
          type: 'text',
          text: content,
          cache_control: buildAnthropicCacheControl(ttl),
        },
      ] as unknown as MessageContentComplex[];
      markerPlaced = true;
    } else {
      continue;
    }

    updatedMessages[i] = cloneMessage(
      originalMessage as MessageWithContent,
      workingContent
    ) as T;
  }

  return updatedMessages;
}

function getMessageRole(message: MessageWithContent): string | undefined {
  if (message instanceof BaseMessage) {
    return message.getType();
  }
  if ('role' in message && typeof message.role === 'string') {
    return message.role;
  }
  return undefined;
}

const SKILL_MESSAGE_SOURCE = 'skill';

/**
 * Synthetic skill/meta messages (reconstructed skill bodies, primed SKILL.md
 * instructions) are re-injected every turn and are not stable conversation
 * turns. They must not anchor a fresh prompt-cache marker — doing so pins the
 * cache to a volatile/duplicated prefix. Stale markers are still stripped from
 * them; only the *adding* of new markers is suppressed. Detected via
 * `additional_kwargs.isMeta === true` or `additional_kwargs.source === 'skill'`.
 */
function isSyntheticMetaMessage(message: MessageWithContent): boolean {
  const { additional_kwargs: kwargs } = message as {
    additional_kwargs?: { isMeta?: unknown; source?: unknown };
  };
  if (kwargs == null) {
    return false;
  }
  return kwargs.isMeta === true || kwargs.source === SKILL_MESSAGE_SOURCE;
}

function isCacheableConversationMessage(message: MessageWithContent): boolean {
  const role = getMessageRole(message);
  return (
    role === 'human' || role === 'user' || role === 'ai' || role === 'assistant'
  );
}

function isAssistantConversationMessage(message: MessageWithContent): boolean {
  const role = getMessageRole(message);
  return role === 'ai' || role === 'assistant';
}

function hasCacheMarker(message: MessageWithContent): boolean {
  return (
    Array.isArray(message.content) &&
    message.content.some((block) => 'cache_control' in block)
  );
}

function addCacheControlToRecentMessages<
  T extends AnthropicMessage | BaseMessage,
>(
  messages: T[],
  maxCachePoints: number,
  canUseMessage: (message: MessageWithContent) => boolean,
  ttl?: PromptCacheTtl
): T[] {
  if (
    !Array.isArray(messages) ||
    messages.length === 0 ||
    maxCachePoints <= 0
  ) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let cachePointsAdded = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;
    const hasArrayContent = Array.isArray(content);
    const canAddCache =
      cachePointsAdded < maxCachePoints &&
      canUseMessage(originalMessage) &&
      !isSyntheticMetaMessage(originalMessage);

    if (!canAddCache && !hasArrayContent) {
      continue;
    }

    let workingContent: MessageContentComplex[];
    let modified = false;

    if (hasArrayContent) {
      const src = content as MessageContentComplex[];
      workingContent = [];
      let lastNonEmptyTextIndex = -1;

      for (let j = 0; j < src.length; j++) {
        const block = src[j];
        if (isCachePoint(block)) {
          modified = true;
          continue;
        }

        const cloned = { ...block };
        if ('cache_control' in cloned) {
          delete (cloned as Record<string, unknown>).cache_control;
          modified = true;
        }

        if ('type' in cloned && cloned.type === 'text') {
          const text = (cloned as { text?: string }).text;
          if (text != null && text.trim() !== '') {
            lastNonEmptyTextIndex = workingContent.length;
          }
        }
        workingContent.push(cloned as MessageContentComplex);
      }

      if (canAddCache && lastNonEmptyTextIndex >= 0) {
        (
          workingContent[lastNonEmptyTextIndex] as Anthropic.TextBlockParam
        ).cache_control = buildAnthropicCacheControl(ttl);
        cachePointsAdded++;
        modified = true;
      }

      if (!modified) {
        continue;
      }
    } else if (
      typeof content === 'string' &&
      content.trim() !== '' &&
      canAddCache
    ) {
      workingContent = [
        {
          type: 'text',
          text: content,
          cache_control: buildAnthropicCacheControl(ttl),
        },
      ] as unknown as MessageContentComplex[];
      cachePointsAdded++;
    } else {
      continue;
    }

    updatedMessages[i] = cloneMessage(
      originalMessage as MessageWithContent,
      workingContent
    ) as T;
  }

  return updatedMessages;
}

export function addCacheControlToStablePrefixMessages<
  T extends AnthropicMessage | BaseMessage,
>(messages: T[], maxCachePoints: number, ttl?: PromptCacheTtl): T[] {
  const assistantMarked = addCacheControlToRecentMessages(
    messages,
    maxCachePoints,
    isAssistantConversationMessage,
    ttl
  );

  if (assistantMarked.some(hasCacheMarker)) {
    return assistantMarked;
  }

  return addCacheControlToRecentMessages(
    messages,
    maxCachePoints,
    isCacheableConversationMessage,
    ttl
  );
}

/**
 * Checks if a message's content has Anthropic cache_control fields.
 */
function hasAnthropicCacheControl(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    const block = content[i];
    if (typeof block !== 'object' || block == null || isProxy(block)) {
      continue;
    }
    try {
      const cacheControl = Object.getOwnPropertyDescriptor(
        block,
        'cache_control'
      );
      if (cacheControl?.enumerable === true) {
        return true;
      }
    } catch {
      continue;
    }
  }
  return false;
}

/**
 * Removes all Anthropic cache_control fields from messages
 * Used when switching from Anthropic to Bedrock provider
 * Returns a new array - only clones messages that require modification.
 */
export function stripAnthropicCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;

    if (!Array.isArray(content) || !hasAnthropicCacheControl(content)) {
      continue;
    }

    const clonedContent = deepCloneContent(content, 'cache_control');
    updatedMessages[i] = cloneMessage(originalMessage, clonedContent);
  }

  return updatedMessages;
}

/**
 * Checks if a message's content has Bedrock cachePoint blocks.
 */
function hasBedrockCachePoint(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    if (isCachePoint(content[i])) return true;
  }
  return false;
}

/**
 * Removes all Bedrock cachePoint blocks from messages
 * Used when switching from Bedrock to Anthropic provider
 * Returns a new array - only clones messages that require modification.
 */
export function stripBedrockCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;

    if (!Array.isArray(content) || !hasBedrockCachePoint(content)) {
      continue;
    }

    const clonedContent = deepCloneContent(content).filter(
      (block) => !isCachePoint(block as MessageContentComplex)
    );
    updatedMessages[i] = cloneMessage(originalMessage, clonedContent);
  }

  return updatedMessages;
}

/**
 * Adds Bedrock Converse API cache points to the latest two user messages.
 * Inserts `{ cachePoint: { type: 'default' } }` as a separate content block
 * immediately after the last text block in each targeted message.
 * Strips ALL existing cache control (both Bedrock and Anthropic formats) from all messages,
 * then adds fresh cache points to the latest two non-tool user messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * Returns a new array - only clones messages that require modification.
 * @param messages - The array of message objects.
 * @returns - A new array of message objects with cache points added.
 */
export function addBedrockCacheControl<
  T extends MessageWithContent & { getType?: () => string; role?: string },
>(messages: T[], ttl?: PromptCacheTtl): T[] {
  if (!Array.isArray(messages) || messages.length === 0) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let cachePointsAdded = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const messageType =
      'getType' in originalMessage &&
      typeof originalMessage.getType === 'function'
        ? originalMessage.getType()
        : undefined;
    const messageRole =
      'role' in originalMessage && typeof originalMessage.role === 'string'
        ? originalMessage.role
        : undefined;

    const isSystemMessage =
      messageType === 'system' || messageRole === 'system';
    if (isSystemMessage) {
      updatedMessages[i] = sanitizeBedrockSystemMessage(originalMessage, ttl);
      continue;
    }

    const isToolMessage = messageType === 'tool' || messageRole === 'tool';
    const isUserMessage = messageType === 'human' || messageRole === 'user';
    const content = originalMessage.content;
    const hasSerializationProps =
      'lc_kwargs' in originalMessage ||
      'lc_serializable' in originalMessage ||
      'lc_namespace' in originalMessage;
    const hasArrayContent = Array.isArray(content);
    const isEmptyString = typeof content === 'string' && content === '';
    const needsCacheAdd =
      cachePointsAdded < 2 &&
      isUserMessage &&
      !isToolMessage &&
      !isEmptyString &&
      !isSyntheticMetaMessage(originalMessage) &&
      (typeof content === 'string' || hasArrayContent);

    if (!needsCacheAdd && !hasArrayContent && !hasSerializationProps) {
      continue;
    }

    let workingContent: string | MessageContentComplex[];
    let modified = hasSerializationProps;

    if (hasArrayContent) {
      // Single pass: clone blocks, strip cache markers, find last
      // non-empty text block for cache point insertion — all at once.
      const src = content as MessageContentComplex[];
      workingContent = [];
      let lastNonEmptyTextIndex = -1;
      for (let j = 0; j < src.length; j++) {
        const block = src[j];
        if (isCachePoint(block)) {
          modified = true;
          continue;
        }
        const cloned = { ...block };
        if ('cache_control' in cloned) {
          delete (cloned as Record<string, unknown>).cache_control;
          modified = true;
        }
        const type = (cloned as { type?: string }).type;
        if (type === ContentTypes.TEXT || type === 'text') {
          const text = (cloned as { text?: string }).text;
          if (text != null && text.trim() !== '') {
            lastNonEmptyTextIndex = workingContent.length;
          }
        }
        workingContent.push(cloned as MessageContentComplex);
      }

      if (!modified && !needsCacheAdd) {
        continue;
      }

      // Insert cache point after the last non-empty text block.
      // Skip if no cacheable text content exists (whitespace-only messages).
      if (needsCacheAdd && lastNonEmptyTextIndex >= 0) {
        workingContent.splice(lastNonEmptyTextIndex + 1, 0, {
          cachePoint: buildBedrockCachePoint(ttl),
        } as MessageContentComplex);
        cachePointsAdded++;
      }
    } else if (typeof content === 'string' && needsCacheAdd) {
      workingContent = [
        { type: ContentTypes.TEXT, text: content },
        { cachePoint: buildBedrockCachePoint(ttl) } as MessageContentComplex,
      ];
      cachePointsAdded++;
    } else if (typeof content === 'string' && hasSerializationProps) {
      workingContent = content;
    } else {
      continue;
    }

    updatedMessages[i] = cloneMessage(originalMessage, workingContent);
  }

  return updatedMessages;
}

/** Bedrock accepts at most this many cachePoints per request across tools, system and messages. */
export const BEDROCK_MAX_CACHE_POINTS = 4;

/**
 * Message cachePoints that fit beside the system checkpoints already in
 * `messages[0]` and one checkpoint reserved for the tool list: 2 when they fit,
 * otherwise 1 (the tail).
 */
export function getBedrockMessageCachePointBudget(
  messages: ReadonlyArray<
    MessageWithContent & { getType?: () => string; role?: string }
  >
): number {
  const first = messages[0] as
    | (MessageWithContent & { getType?: () => string; role?: string })
    | undefined;
  const isSystem =
    first != null &&
    (first.role === 'system' ||
      (typeof first.getType === 'function' && first.getType() === 'system'));
  const systemContent = isSystem ? first.content : undefined;
  const systemCachePoints = Array.isArray(systemContent)
    ? systemContent.filter((block) => isCachePoint(block)).length
    : 0;
  return BEDROCK_MAX_CACHE_POINTS - systemCachePoints - 1 >= 2 ? 2 : 1;
}

/**
 * Bedrock Converse API: single tail cache breakpoint (default strategy).
 *
 * The Bedrock counterpart of {@link addTailCacheControl}. Strips ALL existing
 * cache control (Bedrock cache points and Anthropic `cache_control`) from every
 * message, then inserts exactly ONE `{ cachePoint: { type: 'default' } }` block
 * immediately after the last non-empty text block of the most recent
 * non-synthetic, non-system message. Anchoring on the rolling tail keeps the
 * cached prefix append-only as the conversation grows, instead of re-writing
 * large spans every turn with the legacy "last two user messages" cache points.
 *
 * System messages are sanitized (Anthropic `cache_control` stripped) but never
 * anchored. Synthetic skill/meta messages are skipped as anchors so their
 * volatile content cannot pin the cache.
 *
 * With `maxCachePoints` of 2, the same pass also anchors the latest real human
 * message before the tail. Bedrock only looks back about 20 content blocks from
 * a cachePoint, so after a tool-heavy turn the previous turn's cached prefix is
 * out of reach of the tail alone; the human message that opened it still ends
 * exactly where an earlier tail cached it.
 *
 * Returns a new array - only clones messages that require modification.
 */
export function addBedrockTailCacheControl<
  T extends MessageWithContent & { getType?: () => string; role?: string },
>(messages: T[], ttl?: PromptCacheTtl, maxCachePoints = 1): T[] {
  if (!Array.isArray(messages) || messages.length === 0) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let cachePointsPlaced = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const messageType =
      'getType' in originalMessage &&
      typeof originalMessage.getType === 'function'
        ? originalMessage.getType()
        : undefined;
    const messageRole =
      'role' in originalMessage && typeof originalMessage.role === 'string'
        ? originalMessage.role
        : undefined;

    const isSystemMessage =
      messageType === 'system' || messageRole === 'system';
    if (isSystemMessage) {
      updatedMessages[i] = sanitizeBedrockSystemMessage(originalMessage, ttl);
      continue;
    }

    const content = originalMessage.content;
    const hasSerializationProps =
      'lc_kwargs' in originalMessage ||
      'lc_serializable' in originalMessage ||
      'lc_namespace' in originalMessage;
    const hasArrayContent = Array.isArray(content);
    const isEmptyString = typeof content === 'string' && content === '';
    const isComputerOutput =
      originalMessage instanceof BaseMessage &&
      hasComputerCallOutputMarker(originalMessage);
    const isHumanMessage = messageType === 'human' || messageRole === 'user';
    const canPlaceCachePoint =
      (cachePointsPlaced === 0 ||
        (cachePointsPlaced < maxCachePoints && isHumanMessage)) &&
      !isEmptyString &&
      !isSyntheticMetaMessage(originalMessage) &&
      !isComputerOutput &&
      (typeof content === 'string' || hasArrayContent);

    if (!canPlaceCachePoint && !hasArrayContent && !hasSerializationProps) {
      continue;
    }

    let workingContent: string | MessageContentComplex[];
    let modified = hasSerializationProps;

    if (hasArrayContent) {
      const src = content as MessageContentComplex[];
      workingContent = [];
      let lastNonEmptyTextIndex = -1;
      for (let j = 0; j < src.length; j++) {
        const block = src[j];
        if (isCachePoint(block)) {
          modified = true;
          continue;
        }
        const cloned = { ...block };
        if ('cache_control' in cloned) {
          delete (cloned as Record<string, unknown>).cache_control;
          modified = true;
        }
        const type = (cloned as { type?: string }).type;
        if (type === ContentTypes.TEXT || type === 'text') {
          const text = (cloned as { text?: string }).text;
          if (text != null && text.trim() !== '') {
            lastNonEmptyTextIndex = workingContent.length;
          }
        }
        workingContent.push(cloned as MessageContentComplex);
      }

      if (!modified && !canPlaceCachePoint) {
        continue;
      }

      if (canPlaceCachePoint && lastNonEmptyTextIndex >= 0) {
        workingContent.splice(lastNonEmptyTextIndex + 1, 0, {
          cachePoint: buildBedrockCachePoint(ttl),
        } as MessageContentComplex);
        cachePointsPlaced++;
        modified = true;
      }
    } else if (typeof content === 'string' && canPlaceCachePoint) {
      workingContent = [
        { type: ContentTypes.TEXT, text: content },
        { cachePoint: buildBedrockCachePoint(ttl) } as MessageContentComplex,
      ];
      cachePointsPlaced++;
    } else if (typeof content === 'string' && hasSerializationProps) {
      workingContent = content;
    } else {
      continue;
    }

    updatedMessages[i] = cloneMessage(originalMessage, workingContent);
  }

  return updatedMessages;
}

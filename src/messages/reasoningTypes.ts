import type { ResponseReasoningItem } from 'openai/resources/responses/responses';
import type {
  ThinkingContentText,
  ReasoningContentText,
  GoogleReasoningContentText,
} from '@/types/stream';
import type { MessageContentReasoningBlock } from '@/llm/bedrock/types';

const REASONING_CONTENT_BLOCK_TYPES: ReadonlySet<string> = new Set([
  'think',
  'thinking',
  'thinking_delta',
  'redacted_thinking',
  'reasoning',
  'reasoning-delta',
  'reasoning_content',
]);

/** Identifies every persisted or streamed reasoning content-block shape. */
export function isReasoningContentBlock(block: { type?: unknown }): boolean {
  return (
    typeof block.type === 'string' &&
    REASONING_CONTENT_BLOCK_TYPES.has(block.type)
  );
}

export function isAnthropicThinkingContentBlock(block: {
  type?: unknown;
}): boolean {
  return block.type === 'thinking';
}

type ReasoningPayload = Partial<
  Pick<ThinkingContentText, 'thinking' | 'signature'> &
    Pick<ReasoningContentText, 'think'> &
    Pick<GoogleReasoningContentText, 'reasoning'> &
    Omit<MessageContentReasoningBlock, 'type'> &
    Pick<ResponseReasoningItem, 'summary' | 'content' | 'encrypted_content'>
> & { text?: string; data?: string };

function hasMeaningfulText(value: unknown): boolean {
  return typeof value === 'string' && value.trim() !== '';
}

function hasMeaningfulOpaqueData(value: unknown): boolean {
  return (
    hasMeaningfulText(value) ||
    (value instanceof Uint8Array && value.byteLength > 0)
  );
}

function hasMeaningfulTextPart(value: unknown): boolean {
  return (
    value != null &&
    typeof value === 'object' &&
    'text' in value &&
    hasMeaningfulText(value.text)
  );
}

/** Checks fulfilled provider payloads, not IDs, statuses or empty markers. */
export function hasMeaningfulReasoningPayload(value: unknown): boolean {
  if (typeof value === 'string') {
    return hasMeaningfulText(value);
  }
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const payload = value as ReasoningPayload;
  return (
    hasMeaningfulText(payload.think) ||
    hasMeaningfulText(payload.thinking) ||
    hasMeaningfulText(payload.reasoning) ||
    hasMeaningfulText(payload.text) ||
    hasMeaningfulText(payload.signature) ||
    hasMeaningfulText(payload.data) ||
    hasMeaningfulText(payload.encrypted_content) ||
    hasMeaningfulOpaqueData(payload.redactedContent) ||
    hasMeaningfulText(payload.reasoningText?.text) ||
    hasMeaningfulText(payload.reasoningText?.signature) ||
    (Array.isArray(payload.summary) &&
      payload.summary.some(hasMeaningfulTextPart)) ||
    (Array.isArray(payload.content) &&
      payload.content.some(hasMeaningfulTextPart))
  );
}

/** Reasoning role classification alone is not evidence of generated reasoning. */
export function hasMeaningfulReasoningContentBlock(block: {
  type?: unknown;
}): boolean {
  return isReasoningContentBlock(block) && hasMeaningfulReasoningPayload(block);
}

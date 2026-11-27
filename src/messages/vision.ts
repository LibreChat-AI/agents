import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import { cloneMessage, getCacheControl, withCacheControl } from './cache';
import { flattenLegacyContent, isLegacyConvertible } from './content';
import { hasComputerCallOutputMarker } from '@/utils/toolContent';
import { getMediaMimeType } from '@/utils/tokens';

/** Stands in for each image a model without image input cannot read. */
export const OMITTED_IMAGE_TEXT =
  '[Image omitted because this model does not accept image input.]';

const IMAGE_BLOCK_TYPES = new Set(['image', 'image_url', 'input_image']);
const MIME_TYPED_BLOCK_TYPES = new Set(['file', 'media']);

/** Whether a content block carries an image, in any provider's shape. */
export function isImageContentBlock(block: MessageContentComplex): boolean {
  const type = block.type;
  if (typeof type !== 'string') {
    return false;
  }
  if (IMAGE_BLOCK_TYPES.has(type) || type.startsWith('image/')) {
    return true;
  }
  if (!MIME_TYPED_BLOCK_TYPES.has(type)) {
    return false;
  }
  return (
    getMediaMimeType(block as Record<string, unknown>)?.startsWith('image/') ===
    true
  );
}

/**
 * Replaces every image block with {@link OMITTED_IMAGE_TEXT}, one marker per
 * image, so a model without image input still learns an image was there.
 * Copy-on-write: unchanged messages keep their identity, changed ones are
 * cloned, and a replaced block's prompt-cache breakpoint moves to its marker.
 * With `legacyContent`, a message left with only text is flattened to a
 * string, as `formatContentStrings` does for legacy-content agents.
 * A native computer-call output keeps its screenshot: the Responses API
 * accepts no other output for it, and other providers get it as text later.
 */
export function replaceImageContent(
  messages: BaseMessage[],
  legacyContent = false
): BaseMessage[] {
  let projected: BaseMessage[] | undefined;
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    if (
      !Array.isArray(message.content) ||
      hasComputerCallOutputMarker(message)
    ) {
      continue;
    }
    const source = message.content as MessageContentComplex[];
    let content: MessageContentComplex[] | undefined;
    for (let blockIndex = 0; blockIndex < source.length; blockIndex++) {
      const block = source[blockIndex];
      if (!isImageContentBlock(block)) {
        content?.push(block);
        continue;
      }
      content ??= source.slice(0, blockIndex);
      const marker = getCacheControl(block);
      const text: MessageContentComplex = {
        type: 'text',
        text: OMITTED_IMAGE_TEXT,
      };
      content.push(marker == null ? text : withCacheControl(text, marker));
    }
    if (content == null) {
      continue;
    }
    projected ??= [...messages];
    projected[messageIndex] = cloneMessage(
      message,
      legacyContent && isLegacyConvertible(message, content)
        ? flattenLegacyContent(content)
        : content
    );
  }
  return projected ?? messages;
}

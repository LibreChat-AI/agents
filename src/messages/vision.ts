import type { MessageContentComplex } from '@langchain/core/messages';
import { getCacheControl, withCacheControl } from './cache';
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

/** The text marker standing in for `image`, carrying its prompt-cache breakpoint. */
export function toOmittedImageBlock(
  image: MessageContentComplex
): MessageContentComplex {
  const text: MessageContentComplex = {
    type: 'text',
    text: OMITTED_IMAGE_TEXT,
  };
  const marker = getCacheControl(image);
  return marker == null ? text : withCacheControl(text, marker);
}

/**
 * `block` with its image replaced by the marker, or a native `tool_result`
 * with the images in its `content` replaced; `block` itself when it holds no
 * image.
 */
export function omitImageBlock(
  block: MessageContentComplex
): MessageContentComplex {
  if (isImageContentBlock(block)) {
    return toOmittedImageBlock(block);
  }
  if (block.type !== 'tool_result' || !Array.isArray(block.content)) {
    return block;
  }
  const nested = omitImageBlocks(block.content as MessageContentComplex[]);
  return nested == null ? block : { ...block, content: nested };
}

/** `content` with every image replaced by its marker; `undefined` when it holds none. */
export function omitImageBlocks(
  content: MessageContentComplex[]
): MessageContentComplex[] | undefined {
  let omitted: MessageContentComplex[] | undefined;
  for (let index = 0; index < content.length; index++) {
    const block = content[index];
    const replacement = omitImageBlock(block);
    if (replacement === block) {
      omitted?.push(block);
      continue;
    }
    omitted ??= content.slice(0, index);
    omitted.push(replacement);
  }
  return omitted;
}

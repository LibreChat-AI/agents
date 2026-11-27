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

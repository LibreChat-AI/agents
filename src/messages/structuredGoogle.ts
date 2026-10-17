import type { MessageContent } from '@langchain/core/messages';
import type { MessageContentComplex } from '@/types/stream';
import { ContentTypes } from '@/common';

/** Google tool payloads and persisted native parts retain their structured ordering. */
export function isStructuredGoogleContentPart(
  part: MessageContent[number] | MessageContentComplex
): boolean {
  return (
    typeof part === 'object' &&
    (part.type === 'toolCall' ||
      part.type === 'toolResponse' ||
      part.type === ContentTypes.IMAGE_FILE ||
      part.native_media != null)
  );
}

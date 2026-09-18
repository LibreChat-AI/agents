import { jest } from '@jest/globals';
import type {
  NativeMediaPort,
  NativeMediaPart,
  NativeMediaContent,
} from '@/index';

export const imageData = 'aW1tdXRhYmxlLWltYWdlLWJ5dGVz';
export const imagePart = {
  inlineData: { mimeType: 'image/png', data: imageData },
  thoughtSignature: 'private-signature',
};

export function fixturePort(
  overrides: Partial<NativeMediaPort> = {}
): NativeMediaPort {
  const parts = new Map<string, NativeMediaPart>();
  return {
    start: jest.fn(async () => ({ responseModalities: ['TEXT', 'IMAGE'] })),
    part: jest.fn(
      async ({
        modelRunId,
        part,
        chunkIndex,
        partIndex,
      }: Parameters<
        NativeMediaPort['part']
      >[0]): Promise<NativeMediaContent> => {
        const continuationRef = `${modelRunId}-${chunkIndex}-${partIndex}`;
        parts.set(continuationRef, part);
        if (part.kind === 'text')
          return {
            type: 'text',
            text: part.text,
            native_media: { continuationRef },
          };
        return {
          type: 'image_file',
          image_file: {
            file_id: continuationRef,
            filepath: `/images/owner/${continuationRef}.png`,
            filename: `${continuationRef}.png`,
            type: part.mimeType,
            bytes: 21,
          },
          native_media: { continuationRef },
        };
      }
    ),
    complete: jest.fn(async () => undefined),
    fail: jest.fn(async () => undefined),
    restore: jest.fn(async ({ continuationRef }) => {
      const part = parts.get(continuationRef ?? '');
      if (!part) throw new Error('Missing continuation');
      return part;
    }),
    ...overrides,
  };
}

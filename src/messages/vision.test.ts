import { describe, expect, it } from '@jest/globals';
import type { MessageContentComplex } from '@langchain/core/messages';
import {
  OMITTED_IMAGE_TEXT,
  omitImageBlocks,
  omitArtifactImages,
  isImageContentBlock,
  toOmittedImageBlock,
} from './vision';

const text = { type: 'text', text: 'What is in this picture?' };
const marker = { type: 'text', text: OMITTED_IMAGE_TEXT };
const imageUrl = {
  type: 'image_url',
  image_url: { url: 'data:image/png;base64,AA==', detail: 'auto' },
};

describe('isImageContentBlock', () => {
  it.each<[string, MessageContentComplex]>([
    ['image_url', imageUrl],
    [
      'a standard image block',
      {
        type: 'image',
        source_type: 'base64',
        mime_type: 'image/png',
        data: 'AA==',
      },
    ],
    [
      'an Anthropic image block',
      {
        type: 'image',
        source: { type: 'base64', media_type: 'image/jpeg', data: 'AA==' },
      },
    ],
    [
      'a Responses input_image',
      { type: 'input_image', image_url: 'https://x/y.png' },
    ],
    ['a block typed by its MIME type', { type: 'image/png', data: 'AA==' }],
    [
      'an image media block',
      { type: 'media', mimeType: 'image/webp', data: 'AA==' },
    ],
    [
      'an image file block',
      {
        type: 'file',
        source_type: 'base64',
        mime_type: 'Image/PNG; q=1',
        data: 'AA==',
      },
    ],
  ])('recognizes %s', (_name, block) => {
    expect(isImageContentBlock(block)).toBe(true);
  });

  it.each<[string, MessageContentComplex]>([
    ['text', text],
    [
      'a PDF file block',
      {
        type: 'file',
        source_type: 'base64',
        mime_type: 'application/pdf',
        data: 'AA==',
      },
    ],
    [
      'an audio media block',
      { type: 'media', mimeType: 'audio/wav', data: 'AA==' },
    ],
    ['a file block without a MIME type', { type: 'file', file_id: 'file-1' }],
  ])('leaves %s alone', (_name, block) => {
    expect(isImageContentBlock(block)).toBe(false);
  });
});

describe('omitImageBlocks', () => {
  it('returns undefined when no block holds an image', () => {
    expect(omitImageBlocks([text])).toBeUndefined();
  });

  it('replaces top-level images and images inside a native tool_result', () => {
    const toolResult = {
      type: 'tool_result',
      tool_use_id: 'call-1',
      content: [text, imageUrl],
    };

    expect(omitImageBlocks([imageUrl, toolResult])).toEqual([
      marker,
      { ...toolResult, content: [text, marker] },
    ]);
    expect(toolResult.content).toEqual([text, imageUrl]);
  });

  it('replaces the images of an Anthropic document whose source is custom content', () => {
    const image = {
      type: 'image',
      source: { type: 'base64', media_type: 'image/png', data: 'AA==' },
    };
    const document = {
      type: 'document',
      title: 'Quarterly chart',
      source: { type: 'content', content: [text, image] },
    };

    expect(omitImageBlocks([document])).toEqual([
      { ...document, source: { type: 'content', content: [text, marker] } },
    ]);
    expect(document.source.content).toEqual([text, image]);
  });

  it('leaves a PDF document alone', () => {
    const pdf = {
      type: 'document',
      source: { type: 'base64', media_type: 'application/pdf', data: 'AA==' },
    };

    expect(omitImageBlocks([pdf])).toBeUndefined();
  });
});

describe('omitArtifactImages', () => {
  it('replaces the images of a tool artifact without changing it', () => {
    const artifact = { content: [imageUrl, text], structuredContent: {} };

    expect(omitArtifactImages(artifact)).toEqual({
      content: [marker, text],
      structuredContent: {},
    });
    expect(artifact.content).toEqual([imageUrl, text]);
  });

  it('returns undefined for an artifact without images or content', () => {
    expect(omitArtifactImages({ content: [text] })).toBeUndefined();
    expect(omitArtifactImages({ content: 'plain' })).toBeUndefined();
    expect(omitArtifactImages(undefined)).toBeUndefined();
  });
});

describe('toOmittedImageBlock', () => {
  it('stands in for an image with the marker text', () => {
    expect(toOmittedImageBlock(imageUrl)).toEqual(marker);
  });

  it('keeps the prompt-cache breakpoint of the image on the marker', () => {
    const cacheControl = { type: 'ephemeral' };

    expect(
      toOmittedImageBlock({ ...imageUrl, cache_control: cacheControl })
    ).toEqual({ ...marker, cache_control: cacheControl });
  });
});

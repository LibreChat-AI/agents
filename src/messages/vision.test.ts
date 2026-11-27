import { describe, expect, it } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { MessageContentComplex } from '@langchain/core/messages';
import {
  OMITTED_IMAGE_TEXT,
  isImageContentBlock,
  replaceImageContent,
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

describe('replaceImageContent', () => {
  it('returns the same array when no message carries an image', () => {
    const messages = [
      new HumanMessage('plain'),
      new HumanMessage({ content: [text] }),
      new AIMessage('answer'),
    ];

    expect(replaceImageContent(messages)).toBe(messages);
  });

  it('puts one marker where each image was and keeps everything else in order', () => {
    const untouched = new HumanMessage('earlier turn');
    const source = new HumanMessage({ content: [imageUrl, text, imageUrl] });

    const result = replaceImageContent([untouched, source]);

    expect(result[0]).toBe(untouched);
    expect(result[1]).not.toBe(source);
    expect(result[1]).toBeInstanceOf(HumanMessage);
    expect(result[1].content).toEqual([marker, text, marker]);
    expect(source.content).toEqual([imageUrl, text, imageUrl]);
  });

  it('replaces images in tool results without touching the tool call link', () => {
    const source = new ToolMessage({
      tool_call_id: 'call-1',
      name: 'read_file',
      content: [{ type: 'text', text: 'chart.png' }, imageUrl],
    });

    const [result] = replaceImageContent([source]);

    expect(result).toBeInstanceOf(ToolMessage);
    expect((result as ToolMessage).tool_call_id).toBe('call-1');
    expect(result.content).toEqual([
      { type: 'text', text: 'chart.png' },
      marker,
    ]);
  });

  it('leaves a native computer-call output its screenshot', () => {
    const source = new ToolMessage({
      tool_call_id: 'computer-1',
      content: [
        { type: 'input_image', image_url: 'data:image/png;base64,AA==' },
      ],
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(replaceImageContent([source])[0]).toBe(source);
  });

  it('moves a prompt-cache breakpoint from the image to its marker', () => {
    const cacheControl = { type: 'ephemeral' };
    const source = new HumanMessage({
      content: [text, { ...imageUrl, cache_control: cacheControl }],
    });

    const [result] = replaceImageContent([source]);

    expect(result.content).toEqual([
      text,
      { ...marker, cache_control: cacheControl },
    ]);
  });

  it('flattens a message left with only text for a legacy-content agent', () => {
    const human = new HumanMessage({ content: [text, imageUrl] });
    const tool = new ToolMessage({
      tool_call_id: 'call-1',
      content: [imageUrl],
    });

    const [flatHuman, toolResult] = replaceImageContent([human, tool], true);

    expect(flatHuman.content).toBe(`${text.text}\n${OMITTED_IMAGE_TEXT}`);
    expect(toolResult.content).toEqual([marker]);
  });

  it('keeps array content without a legacy-content agent', () => {
    const [result] = replaceImageContent([
      new HumanMessage({ content: [text, imageUrl] }),
    ]);

    expect(result.content).toEqual([text, marker]);
  });
});

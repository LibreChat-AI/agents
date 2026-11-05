import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { stripImagesFromMessages } from './index';

const imagePart = {
  type: 'image_url' as const,
  image_url: { url: 'data:image/png;base64,AAAA' },
};

describe('stripImagesFromMessages', () => {
  it('returns the input unchanged when visionCapable is true', () => {
    const messages = [
      new HumanMessage({ content: [{ type: 'text', text: 'hi' }, imagePart] }),
    ];
    expect(stripImagesFromMessages(messages, true)).toBe(messages);
  });

  it('strips image_url parts but keeps text for non-vision models', () => {
    const messages = [
      new HumanMessage({
        content: [{ type: 'text', text: 'describe this' }, imagePart],
      }),
    ];
    const result = stripImagesFromMessages(messages, false);
    const content = result[0].content as Array<{ type: string }>;
    expect(content.some((p) => p.type === 'image_url')).toBe(false);
    expect(content.some((p) => p.type === 'text')).toBe(true);
  });

  it('inserts a text placeholder when an image-only message is emptied', () => {
    const messages = [new HumanMessage({ content: [imagePart] })];
    const result = stripImagesFromMessages(messages, false);
    const content = result[0].content as Array<{ type: string }>;
    expect(content.some((p) => p.type === 'image_url')).toBe(false);
    expect(content.length).toBeGreaterThan(0);
    expect(content[0].type).toBe('text');
  });

  it('leaves string content and image-free messages untouched', () => {
    const stringMsg = new HumanMessage({ content: 'plain text' });
    const noImageMsg = new AIMessage({
      content: [{ type: 'text', text: 'ok' }],
    });
    const result = stripImagesFromMessages([stringMsg, noImageMsg], false);
    expect(result[0]).toBe(stringMsg);
    expect(result[1]).toBe(noImageMsg);
  });

  it('does not mutate the original message when stripping', () => {
    const original = new HumanMessage({
      content: [{ type: 'text', text: 'q' }, imagePart],
    });
    stripImagesFromMessages([original], false);
    const content = original.content as Array<{ type: string }>;
    expect(content.some((p) => p.type === 'image_url')).toBe(true);
  });

  /**
   * LangChain standard data content block: how an uploaded image can arrive. These are
   * converted to `image_url` downstream, so they must be stripped here as well - missing
   * this is what let images through on an agent handoff to a text-only model.
   */
  const imageDataBlock = {
    type: 'image' as const,
    source_type: 'base64' as const,
    mime_type: 'image/png',
    data: 'iVBORw0KGgo=',
  };

  it('strips image data content blocks for non-vision models', () => {
    const [msg] = stripImagesFromMessages(
      [
        new HumanMessage({
          content: [{ type: 'text', text: 'describe this' }, imageDataBlock],
        }),
      ],
      false
    );
    expect(msg.content).toEqual([{ type: 'text', text: 'describe this' }]);
  });

  it('keeps data content blocks for vision-capable models', () => {
    const content = [{ type: 'text', text: 'describe this' }, imageDataBlock];
    const [msg] = stripImagesFromMessages(
      [new HumanMessage({ content })],
      true
    );
    expect(msg.content).toEqual(content);
  });
});

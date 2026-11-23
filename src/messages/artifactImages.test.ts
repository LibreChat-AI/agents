import { AIMessageChunk, ToolMessage } from '@langchain/core/messages';
import type {
  MessageContentImageUrl,
  MessageContentComplex,
  BaseMessage,
} from '@langchain/core/messages';
import { projectAnthropicArtifactContent, projectArtifactPayload } from './core';
import { screenshotDataUrl } from '@/utils/__tests__/fixtures/screenshot';
import { HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE } from '@/utils/truncation';
import { ContentTypes } from '@/common';

function screenshotTurn(
  ...urls: string[]
): [AIMessageChunk, ...ToolMessage[]] {
  return [
    new AIMessageChunk({
      content: '',
      tool_calls: urls.map((_, index) => ({
        id: `shot-${index}`,
        name: 'browser_take_screenshot',
        args: {},
        type: 'tool_call' as const,
      })),
    }),
    ...urls.map(
      (url, index) =>
        new ToolMessage({
          content: `Took screenshot ${index}.`,
          tool_call_id: `shot-${index}`,
          artifact: {
            content: [{ type: ContentTypes.IMAGE_URL, image_url: { url } }],
          },
        })
    ),
  ];
}

/** Lengths of the image URLs in provider order, so failures stay readable. */
function imageUrlLengths(messages: BaseMessage[]): number[] {
  const urls: string[] = [];
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content as MessageContentComplex[]) {
      if (block.type !== ContentTypes.IMAGE_URL) {
        continue;
      }
      const imageUrl = (block as MessageContentImageUrl).image_url;
      urls.push(typeof imageUrl === 'string' ? imageUrl : imageUrl.url);
    }
  }
  return urls.map((url) => url.length);
}

/** Provider-visible text of the projected messages, excluding image data. */
function textIn(messages: BaseMessage[]): string {
  const parts: string[] = [];
  for (const message of messages) {
    if (typeof message.content === 'string') {
      parts.push(message.content);
      continue;
    }
    for (const block of message.content as MessageContentComplex[]) {
      if (block.type === ContentTypes.TEXT) {
        parts.push((block as { text: string }).text);
      }
    }
  }
  return parts.join('\n');
}

/** ~900 KB PNG: larger than the 400,000-character tool-result budget. */
const SCREENSHOT = screenshotDataUrl(1_250_000);

describe('tool artifact images', () => {
  describe.each([
    ['OpenAI/Google', projectArtifactPayload],
    ['Anthropic', projectAnthropicArtifactContent],
  ] as const)('%s projection', (_label, project) => {
    it('keeps a screenshot larger than the tool-result character budget', () => {
      const projected = project(screenshotTurn(SCREENSHOT), 400_000);

      expect(imageUrlLengths(projected)).toEqual([SCREENSHOT.length]);
      expect(textIn(projected)).not.toContain('omitted');
    });

    it('keeps a small image under a budget smaller than the image', () => {
      const small = screenshotDataUrl(4_000);
      const projected = project(screenshotTurn(small), 200);

      expect(imageUrlLengths(projected)).toEqual([small.length]);
    });

    it('omits an image above the inline image ceiling', () => {
      const oversized = screenshotDataUrl(HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE + 4);
      const projected = project(screenshotTurn(oversized), 400_000);

      expect(imageUrlLengths(projected)).toEqual([]);
      expect(textIn(projected)).toContain(
        'omitted 1 oversized atomic tool-content block'
      );
    });

    it('still truncates text artifacts to the character budget', () => {
      const messages = screenshotTurn(SCREENSHOT);
      const tool = messages[1];
      tool.artifact = {
        content: [
          { type: ContentTypes.TEXT, text: 'x'.repeat(5_000) },
          { type: ContentTypes.IMAGE_URL, image_url: { url: SCREENSHOT } },
        ],
      };

      const projected = project(messages, 1_000);
      const artifactText = textIn(projected.slice(1));

      expect(imageUrlLengths(projected)).toEqual([SCREENSHOT.length]);
      expect(artifactText).toContain('truncated');
      expect(artifactText.length).toBeLessThanOrEqual(1_100);
    });
  });

  it('shares one inline image budget across an OpenAI/Google tool batch', () => {
    const first = screenshotDataUrl(3_000_000);
    const second = screenshotDataUrl(3_000_000);
    const projected = projectArtifactPayload(
      screenshotTurn(first, second),
      400_000
    );

    expect(imageUrlLengths(projected)).toEqual([first.length]);
    expect(textIn(projected)).toContain(
      'omitted 1 oversized atomic tool-content block'
    );
  });
});

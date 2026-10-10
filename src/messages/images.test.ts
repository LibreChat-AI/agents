import { convertMessagesToCompletionsMessageParams } from '@langchain/openai';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { getProviderSourceMessageIds } from './provenance';
import { projectArtifactPayload } from './core';

const image = (value: string) => ({
  type: 'image_url',
  image_url: { url: `data:image/png;base64,${value.repeat(16)}` },
});

function appendTool(
  messages: BaseMessage[],
  id: string,
  images: ReturnType<typeof image>[] = []
): void {
  messages.push(
    new AIMessage({ content: '', tool_calls: [{ id, name: 'read', args: {} }] })
  );
  messages.push(
    new ToolMessage({
      content: 'read complete',
      tool_call_id: id,
      artifact: { content: images },
      additional_kwargs: { sourceMessageId: id },
    })
  );
}

function imageUrls(messages: BaseMessage[]): string[] {
  return convertMessagesToCompletionsMessageParams({ messages }).flatMap(
    (message) =>
      Array.isArray(message.content)
        ? message.content.flatMap((block) =>
          block.type === 'image_url' ? [block.image_url.url] : []
        )
        : []
  );
}

describe('tool image lifetime', () => {
  it.each([0, 20])(
    'retains images across later tool batches after %i preliminary tools',
    (preliminary) => {
      const messages: BaseMessage[] = [new HumanMessage('inspect the images')];
      for (let i = 0; i < preliminary; i++) appendTool(messages, `prep-${i}`);
      for (const value of ['A', 'B', 'C']) {
        appendTool(messages, value, [image(value)]);
        expect(imageUrls(projectArtifactPayload(messages))).toEqual(
          ['A', 'B', 'C']
            .slice(0, ['A', 'B', 'C'].indexOf(value) + 1)
            .map((v) => image(v).image_url.url)
        );
      }
      appendTool(messages, 'metadata');
      const projected = projectArtifactPayload(messages);
      expect(imageUrls(projected)).toEqual(
        ['A', 'B', 'C'].map((v) => image(v).image_url.url)
      );
      expect(
        projected.slice(messages.length).map(getProviderSourceMessageIds)
      ).toEqual([['A'], ['B'], ['C']]);
      expect(projectArtifactPayload(projected)).toBe(projected);
      expect(
        (messages[messages.length - 3] as ToolMessage).artifact.content
      ).toEqual([image('C')]);
    }
  );

  it('prefers newest images under the aggregate character limit', () => {
    const messages: BaseMessage[] = [new HumanMessage('inspect')];
    appendTool(messages, 'old', [image('A')]);
    appendTool(messages, 'new', [image('B')]);
    appendTool(messages, 'metadata');
    const projected = projectArtifactPayload(messages, 110);
    expect(imageUrls(projected)).toEqual([image('B').image_url.url]);
    expect(
      projected.slice(messages.length).map(getProviderSourceMessageIds)
    ).toEqual([['new']]);
  });

  it('reserves capacity for the current batch before replaying earlier images', () => {
    const messages: BaseMessage[] = [new HumanMessage('inspect')];
    appendTool(messages, 'old', [image('A')]);
    appendTool(messages, 'new', [image('B')]);
    expect(imageUrls(projectArtifactPayload(messages, 130))).toEqual([
      image('B').image_url.url,
    ]);
    expect(imageUrls(projectArtifactPayload(messages, 1_000, false))).toEqual([
      image('B').image_url.url,
    ]);
  });

  it('does not replay prior turns, missing artifacts, malformed images or orphan tools', () => {
    const messages: BaseMessage[] = [new HumanMessage('earlier turn')];
    appendTool(messages, 'old', [image('A')]);
    messages.push(new HumanMessage('new turn'));
    messages.push(
      new ToolMessage({
        content: 'orphan',
        tool_call_id: 'orphan',
        artifact: { content: [image('B')] },
      })
    );
    appendTool(messages, 'malformed', [
      { type: 'image_url', image_url: { url: 'not an image URL' } },
    ]);
    appendTool(messages, 'missing');
    expect(projectArtifactPayload(messages)).toBe(messages);
    expect(imageUrls(projectArtifactPayload(messages))).toEqual([]);
  });

  it('skips oversized images without truncating them or losing smaller older images', () => {
    const messages: BaseMessage[] = [new HumanMessage('inspect')];
    appendTool(messages, 'small', [image('A')]);
    appendTool(messages, 'large', [image('B'.repeat(1_000))]);
    appendTool(messages, 'metadata');
    expect(imageUrls(projectArtifactPayload(messages, 200))).toEqual([
      image('A').image_url.url,
    ]);
  });

  it('does not serialize historical artifact text or execute image serialization hooks', () => {
    let calls = 0;
    const messages: BaseMessage[] = [new HumanMessage('inspect')];
    const unsafeImage = {
      ...image('A'),
      toJSON() {
        calls++;
        return image('A');
      },
    };
    appendTool(messages, 'unsafe', [unsafeImage]);
    (messages[2] as ToolMessage).artifact.content.push({
      type: 'text',
      text: 'historical-artifact-text',
    });
    appendTool(messages, 'metadata');
    expect(projectArtifactPayload(messages)).toBe(messages);
    expect(calls).toBe(0);
  });
});

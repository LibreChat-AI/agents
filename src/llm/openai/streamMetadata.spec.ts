import { expect, test, describe } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { SeenScalarMetadata } from './streamMetadata';
import { dropRepeatedScalarMetadata } from './streamMetadata';

function chunkWithGenerationInfo(
  generationInfo: Record<string, unknown>
): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({ content: '' }),
    generationInfo,
  });
}

describe('dropRepeatedScalarMetadata', () => {
  test.each(['generationInfo', 'response_metadata'] as const)(
    'preserves populated usage after an empty %s snapshot',
    (location) => {
      const seen: SeenScalarMetadata = new Map();
      const usage = {
        prompt_tokens: 21,
        completion_tokens: 39,
        total_tokens: 60,
      };
      const createChunk = (
        snapshot: Record<string, number>
      ): ChatGenerationChunk =>
        new ChatGenerationChunk({
          text: '',
          message: new AIMessageChunk({
            content: '',
            response_metadata:
              location === 'response_metadata' ? { usage: snapshot } : {},
          }),
          generationInfo: {
            completion: 0,
            ...(location === 'generationInfo' && { usage: snapshot }),
          },
        });
      const first = createChunk({});
      const last = createChunk(usage);
      dropRepeatedScalarMetadata(first, seen);
      dropRepeatedScalarMetadata(last, seen);
      const merged = first.concat(last);
      expect(
        location === 'generationInfo'
          ? merged.generationInfo?.usage
          : merged.message.response_metadata.usage
      ).toEqual(usage);
    }
  );

  test('drops a scalar repeated within the same completion (keep-first)', () => {
    const seen: SeenScalarMetadata = new Map();
    const first = chunkWithGenerationInfo({
      completion: 0,
      finish_reason: 'stop',
      model_name: 'm',
    });
    const second = chunkWithGenerationInfo({
      completion: 0,
      finish_reason: 'stop',
      model_name: 'm',
    });

    dropRepeatedScalarMetadata(first, seen);
    dropRepeatedScalarMetadata(second, seen);

    expect(first.generationInfo?.finish_reason).toBe('stop');
    expect(second.generationInfo?.finish_reason).toBeUndefined();
    expect(second.generationInfo?.model_name).toBeUndefined();
  });

  test('keeps a scalar that repeats across different completion indices', () => {
    const seen: SeenScalarMetadata = new Map();
    const choice0 = chunkWithGenerationInfo({
      completion: 0,
      finish_reason: 'stop',
      model_name: 'm',
    });
    const choice1 = chunkWithGenerationInfo({
      completion: 1,
      finish_reason: 'stop',
      model_name: 'm',
    });

    dropRepeatedScalarMetadata(choice0, seen);
    dropRepeatedScalarMetadata(choice1, seen);

    // each completion keeps its own finish metadata (n > 1 aggregates per choice)
    expect(choice0.generationInfo?.finish_reason).toBe('stop');
    expect(choice1.generationInfo?.finish_reason).toBe('stop');
  });

  test('clones before deleting so a shared metadata object is not mutated', () => {
    const seen: SeenScalarMetadata = new Map();
    // Simulate ChatDeepSeek splitting one raw chunk into pieces that share the
    // same generationInfo object.
    const shared = { completion: 0, finish_reason: 'stop', model_name: 'm' };
    const first = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({ content: 'a' }),
      generationInfo: shared,
    });
    const second = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({ content: 'b' }),
      generationInfo: shared,
    });

    dropRepeatedScalarMetadata(first, seen);
    dropRepeatedScalarMetadata(second, seen);

    // the already-emitted first piece (and the shared object) keep finish_reason
    expect(first.generationInfo?.finish_reason).toBe('stop');
    expect(shared.finish_reason).toBe('stop');
    // the repeat piece got its own cleaned copy
    expect(second.generationInfo?.finish_reason).toBeUndefined();
    expect(second.generationInfo).not.toBe(first.generationInfo);
  });

  test('drops repeated usage and provider fields in response_metadata within the same completion', () => {
    const seen: SeenScalarMetadata = new Map();
    const first = new ChatGenerationChunk({
      text: 'Hi',
      message: new AIMessageChunk({
        content: 'Hi',
        response_metadata: {
          provider: 'Google',
          usage: { prompt_tokens: 21, completion_tokens: 39, total_tokens: 60 },
        },
      }),
      generationInfo: { completion: 0 },
    });
    const second = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        response_metadata: {
          provider: 'Google',
          usage: { prompt_tokens: 21, completion_tokens: 39, total_tokens: 60 },
        },
      }),
      generationInfo: { completion: 0 },
    });

    dropRepeatedScalarMetadata(first, seen);
    dropRepeatedScalarMetadata(second, seen);

    expect(first.message.response_metadata.provider).toBe('Google');
    expect(first.message.response_metadata.usage).toEqual({
      prompt_tokens: 21,
      completion_tokens: 39,
      total_tokens: 60,
    });
    expect(second.message.response_metadata.provider).toBeUndefined();
    expect(second.message.response_metadata.usage).toBeUndefined();
  });
});

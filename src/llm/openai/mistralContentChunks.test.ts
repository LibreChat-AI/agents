import { describe, expect, test } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import type { BaseMessageChunk } from '@langchain/core/messages';
import { ChatOpenAI } from './index';

type DeltaConverter = {
  _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: Record<string, unknown>
  ): BaseMessageChunk;
};

const rawResponse = {
  id: 'chatcmpl-1',
  object: 'chat.completion.chunk',
  created: 1,
  model: 'mistral-large-4',
  choices: [],
};

function convertDelta(delta: Record<string, unknown>): AIMessageChunk {
  const model = new ChatOpenAI({
    model: 'mistral-large-4',
    apiKey: 'test',
    configuration: { baseURL: 'https://gateway.example.com/v1' },
  });
  const converter = (model as unknown as { completions: DeltaConverter })
    .completions;
  const message = converter._convertCompletionsDeltaToBaseMessageChunk(
    { role: 'assistant', ...delta },
    rawResponse
  );
  expect(message).toBeInstanceOf(AIMessageChunk);
  return message as AIMessageChunk;
}

function thinking(text: string): Record<string, unknown> {
  return {
    type: 'thinking',
    thinking: [{ type: 'text', text }],
    closed: true,
  };
}

describe('Mistral content chunks on Chat Completions streams', () => {
  test('keeps the first words of the answer that share a chunk with the end of the reasoning', () => {
    const message = convertDelta({
      content: [thinking(' done.'), { type: 'text', text: 'A freelancer' }],
    });

    expect(message.content).toBe('A freelancer');
    expect(message.additional_kwargs.reasoning_content).toBe(' done.');
  });

  test('keeps a tool call that shares a chunk with the reasoning', () => {
    const message = convertDelta({
      content: [thinking('.')],
      tool_calls: [
        {
          index: 0,
          id: 'chatcmpl-tool-1',
          type: 'function',
          function: { name: 'read_page', arguments: '{"page_id": "' },
        },
      ],
    });

    expect(message.content).toBe('');
    expect(message.tool_call_chunks?.[0]).toMatchObject({
      id: 'chatcmpl-tool-1',
      name: 'read_page',
    });
  });

  test('streams a reasoning-only chunk as reasoning with empty text', () => {
    const message = convertDelta({ content: [thinking('The user wants')] });

    expect(message.content).toBe('');
    expect(message.additional_kwargs.reasoning_content).toBe('The user wants');
  });
});

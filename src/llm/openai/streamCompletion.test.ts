import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { describe, expect, test } from '@jest/globals';
import { HumanMessage } from '@langchain/core/messages';
import { ChatOpenAI } from './index';

type StreamFrame = {
  choices: Array<{
    index: number;
    delta: { role?: 'assistant'; content?: string };
    finish_reason: string | null;
  }>;
};

async function startCompletionServer(frames: StreamFrame[]): Promise<{
  baseURL: string;
  close: () => Promise<void>;
}> {
  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    for (const frame of frames) {
      res.write(`data: ${JSON.stringify(frame)}\n\n`);
    }
    if (frames.some((frame) => frame.choices.some((choice) => choice.finish_reason != null))) {
      res.write('data: [DONE]\n\n');
    }
    res.end();
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseURL: `http://127.0.0.1:${port}/v1`,
    close: () => new Promise((resolve, reject) => server.close((error) => error == null ? resolve() : reject(error))),
  };
}

function streamFrame(content: string, finishReason: string | null): StreamFrame {
  return {
    choices: [
      {
        index: 0,
        delta: { role: 'assistant', content },
        finish_reason: finishReason,
      },
    ],
  };
}

describe('ChatOpenAI stream completion', () => {
  test('rejects a partial OpenAI-compatible stream without a terminal finish reason', async () => {
    const server = await startCompletionServer([streamFrame('partial', null)]);
    const model = new ChatOpenAI({
      model: 'test-model',
      apiKey: 'test-key',
      configuration: { baseURL: server.baseURL },
    });
    const received: string[] = [];

    try {
      await expect(async () => {
        for await (const chunk of await model.stream([new HumanMessage('test')])) {
          received.push(String(chunk.content));
        }
      }).rejects.toThrow('OpenAI-compatible stream ended before completion');
      expect(received).toContain('partial');
    } finally {
      await server.close();
    }
  });

  test('accepts a stream with a terminal finish reason', async () => {
    const server = await startCompletionServer([
      streamFrame('complete', null),
      streamFrame('', 'stop'),
    ]);
    const model = new ChatOpenAI({
      model: 'test-model',
      apiKey: 'test-key',
      configuration: { baseURL: server.baseURL },
    });
    const received: string[] = [];

    try {
      for await (const chunk of await model.stream([new HumanMessage('test')])) {
        received.push(String(chunk.content));
      }
      expect(received).toContain('complete');
    } finally {
      await server.close();
    }
  });
});

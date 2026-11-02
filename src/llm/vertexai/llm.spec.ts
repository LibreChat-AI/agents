import { config } from 'dotenv';
config();
import { expect, test, describe, jest } from '@jest/globals';

jest.setTimeout(90000);
import {
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatVertexAI } from './index';

const gemini3Models = [
  'gemini-3-pro-preview',
  'gemini-3-flash-preview',
  'gemini-3.1-flash-lite-preview',
];

const weatherTool = tool(async () => 'The weather is 80 degrees and sunny', {
  name: 'weather',
  description: 'Gets the current weather in a given location',
  schema: z.object({
    location: z.string().describe('The city to get the weather for'),
  }),
});

describe('ChatVertexAI upstream compatibility', () => {
  test('serialization uses the LibreChat constructor name on the Vertex namespace', () => {
    const model = new ChatVertexAI();
    expect(JSON.stringify(model)).toEqual(
      '{"lc":1,"type":"constructor","id":["langchain","chat_models","vertexai","LibreChatVertexAI"],"kwargs":{"platform_type":"gcp"}}'
    );
  });

  test('labels parameter support', () => {
    expect(() => {
      const model = new ChatVertexAI({
        labels: {
          team: 'test',
          environment: 'development',
        },
      });
      expect(model.platform).toEqual('gcp');
    }).not.toThrow();
  });

  test('constructor overload supports model string', () => {
    const model = new ChatVertexAI('gemini-1.5-pro');
    expect(model.model).toEqual('gemini-1.5-pro');
    expect(model.platform).toEqual('gcp');
  });
});

describe.each(gemini3Models)(
  'Vertex AI reasoning with thinkingLevel (%s)',
  (modelName) => {
    const model = new ChatVertexAI({
      model: modelName,
      location: 'global',
      maxRetries: 2,
      thinkingConfig: {
        thinkingLevel: 'HIGH',
        includeThoughts: true,
      },
    });

    test('invoke with thinkingLevel produces a response with reasoning tokens', async () => {
      const result = await model.invoke('What is 2+2? Think step by step.');
      expect(result.content).toBeDefined();
      const reasoningTokens = (result.usage_metadata as Record<string, unknown>)
        ?.output_token_details;
      expect(reasoningTokens).toBeDefined();
      expect(
        (reasoningTokens as Record<string, number>)?.reasoning
      ).toBeGreaterThan(0);
    });

    test('stream: usage_metadata includes reasoning in output_tokens (issue LibreChat#13006)', async () => {
      let finalChunk: AIMessageChunk | undefined;
      for await (const chunk of await model.stream(
        'What is 2+2? Think step by step.'
      )) {
        finalChunk = finalChunk ? finalChunk.concat(chunk) : chunk;
      }
      const usage = finalChunk?.usage_metadata;
      expect(usage).toBeDefined();
      const reasoning = (
        usage as { output_token_details?: { reasoning?: number } }
      )?.output_token_details?.reasoning;
      expect(reasoning).toBeGreaterThan(0);
      expect(usage!.total_tokens).toBe(
        usage!.input_tokens + usage!.output_tokens
      );
    });
  }
);

describe.each(gemini3Models)(
  'Vertex AI tool calling with thought signatures (%s)',
  (modelName) => {
    const model = new ChatVertexAI({
      model: modelName,
      location: 'global',
      maxRetries: 2,
    });
    const modelWithTools = model.bindTools([weatherTool]);

    test('invoke: tool call completes round-trip with thought signature', async () => {
      const result = await modelWithTools.invoke(
        'What is the current weather in San Francisco?'
      );
      expect(result.tool_calls).toBeDefined();
      expect(result.tool_calls!.length).toBeGreaterThanOrEqual(1);
      expect(result.tool_calls![0].id).toBeDefined();

      const toolMessage = new ToolMessage({
        content: 'The weather is 80 degrees and sunny',
        tool_call_id: result.tool_calls![0].id ?? '',
      });

      // Critical round-trip: sending the function call + tool result back to the API.
      // Without proper thought_signature handling, this fails with
      // "function call X is missing a thought_signature"
      const finalResult = await model.invoke([
        new HumanMessage('What is the current weather in San Francisco?'),
        result,
        toolMessage,
      ]);
      expect(finalResult.content).toBeDefined();
    });

    test('stream: tool call completes round-trip with thought signature', async () => {
      let finalChunk: AIMessageChunk | undefined;
      for await (const chunk of await modelWithTools.stream(
        'What is the current weather in San Francisco?'
      )) {
        finalChunk = finalChunk ? finalChunk.concat(chunk) : chunk;
      }
      expect(finalChunk).toBeDefined();
      expect(finalChunk?.tool_calls).toBeDefined();
      expect(finalChunk?.tool_calls!.length).toBeGreaterThanOrEqual(1);

      const toolMessage = new ToolMessage({
        content: 'The weather is 80 degrees and sunny',
        tool_call_id: finalChunk?.tool_calls![0].id ?? '',
      });

      // Round-trip: send tool result back — verifies thought_signature handling
      const finalResult = await model.invoke([
        new HumanMessage('What is the current weather in San Francisco?'),
        finalChunk!,
        toolMessage,
      ]);
      expect(finalResult.content).toBeDefined();
    });
  }
);

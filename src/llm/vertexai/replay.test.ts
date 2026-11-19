import { afterEach, describe, expect, jest, test } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { GeminiPart, GeminiRequest } from '@langchain/google-common';
import type { BaseMessage } from '@langchain/core/messages';
import { restoreGeminiReplayParts } from '@/llm/google/utils/signatures';
import { formatContentStrings } from '@/messages/content';
import { ChatVertexAI } from './index';

function serve(parts: GeminiPart[], split: boolean): Response {
  const frames = (split ? parts.map((part) => [part]) : [parts]).map(
    (parts) => ({
      candidates: [{ index: 0, content: { role: 'model', parts } }],
    })
  );
  return new Response(JSON.stringify(frames), {
    headers: { 'content-type': 'application/json' },
  });
}

async function stream(
  model: ChatVertexAI,
  messages: BaseMessage[]
): Promise<AIMessageChunk> {
  let combined = new AIMessageChunk('');
  for await (const chunk of await model.stream(
    formatContentStrings(messages)
  )) {
    combined = combined.concat(chunk);
  }
  return combined;
}

function withCallIds(
  parts: GeminiPart[],
  message: AIMessageChunk
): GeminiPart[] {
  let index = 0;
  return parts.map((part) => {
    if (!('functionCall' in part)) return part;
    const call = message.tool_calls?.[index++];
    if (call?.id == null) throw new Error('Missing tool-call ID');
    return {
      ...part,
      functionCall: {
        ...part.functionCall,
        id: call.id,
      },
    };
  });
}

function toolResults(message: AIMessageChunk): ToolMessage[] {
  return (message.tool_calls ?? []).map(
    (call) =>
      new ToolMessage({
        tool_call_id: call.id ?? '',
        name: call.name,
        content: JSON.stringify({ value: call.args.value }),
      })
  );
}

afterEach(() => {
  jest.restoreAllMocks();
});

for (const modelName of ['gemini-3.8-flash', 'gemini-4-hydrogen']) {
  describe(`${modelName} reasoning replay`, () => {
    test.each([
      [false, 0, false],
      [false, 25, false],
      [true, 0, false],
      [true, 25, false],
      [false, 0, true],
      [false, 25, true],
      [true, 0, true],
      [true, 25, true],
    ] as const)(
      'keeps signatures with split=%s smoothing=%s sequential=%s @scenario:vertex-reasoning-tools',
      async (split, delay, sequential) => {
        const requests: GeminiRequest[] = [];
        const firstParts: GeminiPart[] = [
          { text: 'Plan.', thought: true, thoughtSignature: 'thought-one' },
          {
            functionCall: { name: 'calculator', args: { value: 1 } },
            thoughtSignature: 'call-one',
          },
          ...(!sequential
            ? [{ functionCall: { name: 'calculator', args: { value: 2 } } }]
            : []),
        ];
        const secondParts: GeminiPart[] = [
          { text: 'Next.', thought: true, thoughtSignature: 'thought-two' },
          {
            functionCall: { name: 'calculator', args: { value: 2 } },
            thoughtSignature: 'call-two',
          },
        ];
        const responses = [
          firstParts,
          ...(sequential ? [secondParts] : []),
          [{ text: 'Done.' }],
        ];
        jest
          .spyOn(globalThis, 'fetch')
          .mockImplementation(async (_input, init) => {
            if (typeof init?.body !== 'string')
              throw new Error('Expected JSON request');
            requests.push(JSON.parse(init.body) as GeminiRequest);
            const parts = responses.shift();
            if (parts == null) throw new Error('Unexpected model request');
            return serve(parts, split);
          });
        const model = new ChatVertexAI({
          model: modelName,
          apiKey: 'test-key',
          location: 'global',
          _lc_stream_delay: delay,
          maxRetries: 0,
          thinkingConfig: { includeThoughts: true },
        });
        const messages: BaseMessage[] = [
          new HumanMessage('Calculate both values.'),
        ];
        const first = await stream(model, messages);
        expect(first.content).toContainEqual({
          type: 'reasoning',
          reasoning: 'Plan.',
        });
        expect(first.tool_calls).toHaveLength(sequential ? 1 : 2);
        const expectedFirst = withCallIds(firstParts, first);
        expect(restoreGeminiReplayParts(first)).toEqual(expectedFirst);
        // Exercise restored history as well as live chunk aggregation.
        messages.push(
          new AIMessage({
            content: structuredClone(first.content),
            additional_kwargs: structuredClone(first.additional_kwargs),
            tool_calls: structuredClone(first.tool_calls),
          }),
          ...toolResults(first)
        );
        if (sequential) {
          const second = await stream(model, messages);
          expect(second.tool_calls).toHaveLength(1);
          expect(requests[1].contents?.[1].parts).toEqual(expectedFirst);
          messages.push(second, ...toolResults(second));
          expect(restoreGeminiReplayParts(second)).toEqual(
            withCallIds(secondParts, second)
          );
        }
        const final = await stream(model, messages);
        expect(final.content).toBe('Done.');
        expect(requests.at(-1)?.contents?.[1].parts).toEqual(expectedFirst);
        const followup = requests.at(-1)?.contents ?? [];
        expect(
          followup.flatMap((content) =>
            content.parts.filter((part) => 'functionResponse' in part)
          )
        ).toHaveLength(2);
        expect(
          followup.flatMap((content) =>
            content.parts.flatMap((part) =>
              'functionCall' in part ? [part.thoughtSignature] : []
            )
          )
        ).toEqual(
          sequential ? ['call-one', 'call-two'] : ['call-one', undefined]
        );
        expect(responses).toHaveLength(0);
      }
    );
  });
}

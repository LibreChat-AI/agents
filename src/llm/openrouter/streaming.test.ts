import { concat } from '@langchain/core/utils/stream';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { AIMessageChunk, BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type { OpenAIClient } from '@langchain/openai';
import { ChatOpenRouter } from './index';

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;
type Binding = 'unbound' | 'bindTools' | 'withConfig';

class InspectableChatOpenRouter extends ChatOpenRouter {
  get delegate() {
    return this.completions;
  }
}

class StreamCallback extends BaseCallbackHandler {
  name = 'openrouter-streaming-regression';
  override awaitHandlers = true;
  lc_prefer_streaming = false;
  message?: BaseMessage;

  override handleLLMEnd(output: LLMResult): void {
    const generation = output.generations.at(0)?.at(0);
    if (generation != null && 'message' in generation) {
      this.message = generation.message as BaseMessage;
    }
  }
}

const modelName = 'anthropic/claude-sonnet-4.5';
const tool = {
  type: 'function' as const,
  function: {
    name: 'noop',
    parameters: { type: 'object', properties: {} },
  },
};
const usage = { prompt_tokens: 21, completion_tokens: 39, total_tokens: 60 };

function rawChunk(
  delta: RawChunk['choices'][number]['delta'],
  finishReason: 'tool_calls' | null = null,
  withUsage = false
): RawChunk {
  return {
    id: 'gen-reasoning',
    object: 'chat.completion.chunk',
    created: 1,
    model: modelName,
    ...{ provider: 'FixtureProvider' },
    choices: [{ index: 0, delta, finish_reason: finishReason, logprobs: null }],
    ...(withUsage ? { usage } : {}),
  };
}

function bindRouter(router: InspectableChatOpenRouter, binding: Binding) {
  if (binding === 'unbound') return router;
  if (binding === 'bindTools') return router.bindTools([tool]);
  return router.withConfig({ tools: [tool], stop: ['END'] });
}

afterEach(() => jest.restoreAllMocks());

describe.each(['unbound', 'bindTools', 'withConfig'] as const)(
  '%s OpenRouter reasoning stream',
  (binding) => {
    test.each(
      (['stream', 'invoke', 'callback-stream'] as const).flatMap((entry) =>
        [true, false].map((indexed) => ({ entry, indexed }))
      )
    )(
      'preserves $entry reasoning (indexed=$indexed) through tool continuation',
      async ({ entry, indexed }) => {
        const router = new InspectableChatOpenRouter({
          model: modelName,
          apiKey: 'offline-test',
          streaming: true,
          _lc_stream_delay: 0,
          reasoning: { effort: 'high', exclude: false },
        });
        const reasoningDetails = [
          {
            type: 'reasoning.text',
            ...(indexed ? { index: 0 } : {}),
            text: 'think',
            format: 'test-v1',
          },
          {
            type: 'reasoning.encrypted',
            ...(indexed ? { index: 1 } : {}),
            id: 'reasoning-signature',
            data: 'encrypted-payload',
            format: 'test-v1',
          },
        ];
        const requests: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming[] =
          [];
        const client: {
          completionWithRetry(
            request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
          ): Promise<AsyncIterable<RawChunk>>;
        } = router.delegate;
        client.completionWithRetry = async (request) => {
          requests.push(request);
          return {
            async *[Symbol.asyncIterator]() {
              yield rawChunk({
                role: 'assistant',
                content: '',
                ...{ reasoning_details: reasoningDetails },
              });
              yield rawChunk({
                content: 'Hi',
                tool_calls: [
                  {
                    index: 0,
                    id: 'call-noop',
                    type: 'function',
                    function: { name: 'noop', arguments: '{}' },
                  },
                ],
              });
              yield rawChunk({ content: '' }, 'tool_calls');
              yield rawChunk({ content: '' }, 'tool_calls', true);
            },
          };
        };
        jest.spyOn(router.delegate, 'getNumTokens').mockResolvedValue(1);
        const target = bindRouter(router, binding);
        expect(target).toBeInstanceOf(InspectableChatOpenRouter);
        const callback = new StreamCallback();
        callback.lc_prefer_streaming = entry === 'callback-stream';
        for (let run = 0; run < 2; run++) {
          let result: AIMessageChunk | undefined;
          if (entry === 'stream') {
            for await (const chunk of await target.stream('hello', {
              callbacks: [callback],
            })) {
              result = result == null ? chunk : concat(result, chunk);
            }
          } else {
            result = await target.invoke('hello', { callbacks: [callback] });
          }
          expect(result).toBeDefined();
          for (const message of [result, callback.message]) {
            expect(message?.content).toBe('Hi');
            expect(message?.additional_kwargs.reasoning_details).toEqual(
              reasoningDetails
            );
            expect(message?.response_metadata).toMatchObject({
              provider: 'FixtureProvider',
              usage,
            });
            if (entry !== 'callback-stream') {
              expect(message?.response_metadata).toMatchObject({
                model_name: modelName,
                finish_reason: 'tool_calls',
              });
            }
          }
          expect(result?.tool_calls).toEqual([
            { id: 'call-noop', name: 'noop', args: {}, type: 'tool_call' },
          ]);
          if (result == null) throw new Error('Missing streamed message');
          await target.invoke([
            new HumanMessage('hello'),
            result,
            new ToolMessage({ tool_call_id: 'call-noop', content: 'done' }),
          ]);
          expect(requests.at(-1)?.messages.at(1)).toMatchObject({
            role: 'assistant',
            content: [
              { type: 'thinking', thinking: 'think' },
              {
                type: 'redacted_thinking',
                data: 'encrypted-payload',
                id: 'reasoning-signature',
              },
            ],
          });
        }
        for (const request of requests) {
          expect(request).toMatchObject({
            reasoning: { effort: 'high', exclude: false },
          });
          if (binding !== 'unbound') expect(request.tools).toHaveLength(1);
          if (binding === 'withConfig') expect(request.stop).toEqual(['END']);
        }
      }
    );
  }
);

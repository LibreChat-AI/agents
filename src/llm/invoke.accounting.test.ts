import { describe, expect, it, jest } from '@jest/globals';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ChatModel } from '@/types';
import { InvalidModelToolCallError } from '@/graphs/acceptedModelResponse';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { OutputTruncationError } from '@/llm/truncation';
import { attemptInvoke } from '@/llm/invoke';
import { Providers } from '@/common';

const usage = { input_tokens: 120, output_tokens: 30, total_tokens: 150 };

function modelReturning(
  mode: 'stream' | 'invoke',
  response: AIMessage | AIMessageChunk
): ChatModel {
  return {
    invoke: async () => response,
    ...(mode === 'stream'
      ? {
        stream: async function* () {
          yield response;
          yield new AIMessageChunk({ content: '' });
        },
      }
      : {}),
  } as ChatModel;
}

describe.each(['stream', 'invoke'] as const)(
  'attemptInvoke completion accounting (%s)',
  (mode) => {
    const Message = mode === 'stream' ? AIMessageChunk : AIMessage;

    it('reports completed Bedrock usage once before rejecting a truncated tool call', async () => {
      const response = new Message({
        content: '',
        tool_calls: [
          { id: 'call-1', name: 'write_file', args: { path: 'out' } },
        ],
        response_metadata:
          mode === 'stream'
            ? { messageStop: { stopReason: 'max_tokens' } }
            : { stopReason: 'max_tokens' },
        usage_metadata: usage,
      });
      const onModelComplete = jest.fn((message: BaseMessage) => {
        expect(message).toMatchObject({ usage_metadata: usage });
      });
      const executeTool = jest.fn();
      const result = attemptInvoke({
        model: modelReturning(mode, response),
        messages: [new HumanMessage('write a file')],
        provider: Providers.BEDROCK,
        onChunk: () => {},
        onModelComplete,
      }).then(executeTool);

      await expect(result).rejects.toBeInstanceOf(OutputTruncationError);
      expect(onModelComplete).toHaveBeenCalledTimes(1);
      expect(executeTool).not.toHaveBeenCalled();
    });

    it('observes a normal completion once before tool-call cleanup without emitting events', async () => {
      const response = new Message({
        content: 'done',
        tool_calls: [{ id: 'unnamed', name: '', args: {} }],
        usage_metadata: usage,
      });
      const onModelComplete = jest.fn((message: BaseMessage) => {
        expect(message).toMatchObject({
          tool_calls: [{ id: 'unnamed', name: '', args: {} }],
          usage_metadata: usage,
        });
      });
      const customEvent = jest.fn(() => {});
      const callbacks = [
        BaseCallbackHandler.fromMethods({ handleCustomEvent: customEvent }),
      ];

      const result = await attemptInvoke(
        {
          model: modelReturning(mode, response),
          messages: [new HumanMessage('hi')],
          provider: Providers.BEDROCK,
          onChunk: () => {},
          onModelComplete,
        },
        { callbacks }
      );

      expect(onModelComplete).toHaveBeenCalledTimes(1);
      expect(result.messages?.[0]).toMatchObject({
        content: 'done',
        tool_calls: [],
        usage_metadata: usage,
      });
      expect(customEvent).not.toHaveBeenCalled();
    });

    it('does not fabricate a completion when the provider request throws', async () => {
      const error = new Error('provider request failed');
      const fail = async (): Promise<never> => {
        throw error;
      };
      const onModelComplete = jest.fn<(message: BaseMessage) => void>();
      const model: ChatModel = {
        invoke: fail,
        ...(mode === 'stream' ? { stream: fail } : {}),
      };

      await expect(
        attemptInvoke({
          model,
          messages: [new HumanMessage('hi')],
          provider: Providers.BEDROCK,
          onChunk: () => {},
          onModelComplete,
        })
      ).rejects.toBe(error);
      expect(onModelComplete).not.toHaveBeenCalled();
    });

    it.each(['getter', 'toJSON', 'proxy'] as const)(
      'rejects unsafe tool arguments without invoking their %s side effects or the observer',
      async (kind) => {
        const sideEffect = jest.fn(() => 'unsafe');
        const response = new Message({
          content: '',
          tool_calls: [{ id: 'call-unsafe', name: 'write_file', args: {} }],
          usage_metadata: usage,
        });
        const args = response.tool_calls![0].args;
        if (kind === 'getter') {
          Object.defineProperty(args, 'path', {
            enumerable: true,
            get: sideEffect,
          });
        }
        if (kind === 'toJSON') {
          args.toJSON = sideEffect;
        }
        if (kind === 'proxy') {
          response.tool_calls![0].args = new Proxy(args, { get: sideEffect });
        }
        const onModelComplete = jest.fn<(message: BaseMessage) => void>();

        await expect(
          attemptInvoke({
            model: modelReturning(mode, response),
            messages: [new HumanMessage('hi')],
            provider: Providers.BEDROCK,
            onChunk: () => {},
            onModelComplete,
          })
        ).rejects.toBeInstanceOf(InvalidModelToolCallError);
        expect(onModelComplete).not.toHaveBeenCalled();
        expect(sideEffect).not.toHaveBeenCalled();
      }
    );
  }
);

describe('attemptInvoke streaming completion accounting', () => {
  it('includes usage delivered after the stop-reason chunk before truncation rejects', async () => {
    const model: ChatModel = {
      invoke: async () => new AIMessageChunk('unused'),
      stream: async () =>
        (async function* () {
          yield new AIMessageChunk({
            content: '',
            tool_call_chunks: [
              { id: 'call-1', name: 'write_file', args: '{"path":', index: 0 },
            ],
          });
          yield new AIMessageChunk({
            content: '',
            response_metadata: { messageStop: { stopReason: 'max_tokens' } },
          });
          yield new AIMessageChunk({ content: '', usage_metadata: usage });
        })(),
    };
    const onModelComplete = jest.fn<(message: BaseMessage) => void>();

    await expect(
      attemptInvoke({
        model,
        messages: [new HumanMessage('write a file')],
        provider: Providers.BEDROCK,
        onChunk: () => {},
        onModelComplete,
      })
    ).rejects.toBeInstanceOf(OutputTruncationError);

    expect(onModelComplete).toHaveBeenCalledTimes(1);
    expect(onModelComplete.mock.calls[0][0]).toMatchObject({
      usage_metadata: usage,
    });
  });

  it('does not fabricate a completion for an empty stream', async () => {
    const model: ChatModel = {
      invoke: async () => new AIMessageChunk('unused'),
      stream: async () => (async function* () {})(),
    };
    const onModelComplete = jest.fn<(message: BaseMessage) => void>();

    await expect(
      attemptInvoke({
        model,
        messages: [new HumanMessage('hi')],
        provider: Providers.BEDROCK,
        onModelComplete,
      })
    ).rejects.toThrow('The model provider returned an empty response.');
    expect(onModelComplete).not.toHaveBeenCalled();
  });

  it('does not report a completed response when cancellation rejects a partial stream', async () => {
    const abort = new AbortController();
    const error = new StreamLimitExceededError({
      kind: 'delta_events',
      limit: 1,
      observed: 2,
    });
    const onModelComplete = jest.fn<(message: BaseMessage) => void>();

    await expect(
      attemptInvoke(
        {
          model: modelReturning(
            'stream',
            new AIMessageChunk({ content: 'partial', usage_metadata: usage })
          ),
          messages: [new HumanMessage('hi')],
          provider: Providers.BEDROCK,
          onChunk: () => {
            abort.abort(error);
          },
          onModelComplete,
        },
        { signal: abort.signal }
      )
    ).rejects.toBe(error);
    expect(onModelComplete).not.toHaveBeenCalled();
  });
});

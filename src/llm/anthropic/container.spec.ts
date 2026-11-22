import { describe, expect, test } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseChatModelCallOptions } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  AnthropicMessageStreamEvent,
  AnthropicRequestOptions,
  AnthropicStreamingMessageCreateParams,
} from './types';
import type { CustomAnthropicCallOptions } from './index';
import { getCurrentTurnContainerId } from './utils/container';
import { CustomAnthropic as ChatAnthropic } from './index';

type ParsedOptions = ChatAnthropic['ParsedCallOptions'];

class RecordingAnthropic extends ChatAnthropic {
  capturedRequest: AnthropicStreamingMessageCreateParams | undefined;

  constructor() {
    super({ apiKey: 'fake-key', model: 'claude-sonnet-4-20250514' });
  }

  protected override async createStreamWithRetry(
    request: AnthropicStreamingMessageCreateParams,
    _options?: AnthropicRequestOptions
  ): ReturnType<ChatAnthropic['createStreamWithRetry']> {
    this.capturedRequest = request;
    const events = streamEvents();
    return {
      controller: { abort: (): void => {} },
      async *[Symbol.asyncIterator](): AsyncGenerator<AnthropicMessageStreamEvent> {
        for (const event of events) {
          yield event;
        }
      },
    } as unknown as Awaited<ReturnType<ChatAnthropic['createStreamWithRetry']>>;
  }
}

function streamEvents(): AnthropicMessageStreamEvent[] {
  return [
    {
      type: 'message_start',
      message: {
        id: 'msg_01ABC',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-sonnet-4-20250514',
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 0 },
      },
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'ok' },
    },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: 1 },
    },
    { type: 'message_stop' },
  ] as unknown as AnthropicMessageStreamEvent[];
}

type StreamChunksModel = {
  _streamChatModelEvents: (
    messages: BaseMessage[],
    options: ParsedOptions
  ) => AsyncGenerator<unknown>;
};

const asOptions = (options: CustomAnthropicCallOptions = {}): ParsedOptions =>
  options as unknown as ParsedOptions;

async function drain(stream: AsyncGenerator<unknown>): Promise<void> {
  for await (const _item of stream) {
    // Consume the stream to trigger request construction.
  }
}

const streamPaths: Array<{
  name: string;
  run: (
    model: RecordingAnthropic,
    messages: BaseMessage[],
    options?: CustomAnthropicCallOptions
  ) => Promise<void>;
}> = [
  {
    name: '_streamResponseChunks',
    run: (model, messages, options) =>
      drain(model._streamResponseChunks(messages, asOptions(options))),
  },
  {
    name: '_streamChatModelEvents',
    run: (model, messages, options) =>
      drain(
        (model as unknown as StreamChunksModel)._streamChatModelEvents(
          messages,
          asOptions(options) as BaseChatModelCallOptions as ParsedOptions
        )
      ),
  },
];

const containerKwargs = (id: string): { container: { id: string } } => ({
  container: { id },
});

const aiWithContainer = (
  id: string,
  extra: Partial<AIMessage> = {}
): AIMessage =>
  new AIMessage({
    content: 'working',
    additional_kwargs: containerKwargs(id),
    ...extra,
  });

const toolCallMessage = (id: string, containerId?: string): AIMessage =>
  new AIMessage({
    content: '',
    tool_calls: [{ id, name: 'mcp_tool', args: {}, type: 'tool_call' }],
    additional_kwargs: containerId != null ? containerKwargs(containerId) : {},
  });

describe.each(streamPaths)('CustomAnthropic.$name container', ({ run }) => {
  test('sends the container of the current turn', async () => {
    const model = new RecordingAnthropic();
    await run(model, [
      new HumanMessage('search and call the tool'),
      toolCallMessage('toolu_1', 'container_cur'),
      new ToolMessage({ content: 'result', tool_call_id: 'toolu_1' }),
    ]);

    expect(model.capturedRequest?.container).toBe('container_cur');
  });

  test('uses the latest container of the current turn', async () => {
    const model = new RecordingAnthropic();
    await run(model, [
      new HumanMessage('go'),
      toolCallMessage('toolu_1', 'container_old'),
      new ToolMessage({ content: 'r1', tool_call_id: 'toolu_1' }),
      toolCallMessage('toolu_2', 'container_new'),
      new ToolMessage({ content: 'r2', tool_call_id: 'toolu_2' }),
    ]);

    expect(model.capturedRequest?.container).toBe('container_new');
  });

  test('does not send a container from an earlier turn', async () => {
    const model = new RecordingAnthropic();
    await run(model, [
      new HumanMessage('first question'),
      aiWithContainer('container_prev'),
      new HumanMessage('second question'),
    ]);

    expect(model.capturedRequest?.container).toBeUndefined();
  });

  test('keeps the container across a message injected inside the turn', async () => {
    const model = new RecordingAnthropic();
    await run(model, [
      new HumanMessage('go'),
      toolCallMessage('toolu_1', 'container_cur'),
      new ToolMessage({ content: 'r1', tool_call_id: 'toolu_1' }),
      new HumanMessage({
        content: 'hook context',
        additional_kwargs: { role: 'user', injected: true, source: 'hook' },
      }),
    ]);

    expect(model.capturedRequest?.container).toBe('container_cur');
  });

  test('keeps the container across a routing cue', async () => {
    const model = new RecordingAnthropic();
    await run(model, [
      new HumanMessage('go'),
      aiWithContainer('container_cur'),
      new HumanMessage({
        content: 'continue',
        additional_kwargs: { role: 'user', isMeta: true, source: 'routing' },
      }),
    ]);

    expect(model.capturedRequest?.container).toBe('container_cur');
  });

  test('prefers an explicit container option', async () => {
    const model = new RecordingAnthropic();
    await run(
      model,
      [
        new HumanMessage('go'),
        toolCallMessage('toolu_1', 'container_cur'),
        new ToolMessage({ content: 'r1', tool_call_id: 'toolu_1' }),
      ],
      { container: 'container_explicit' }
    );

    expect(model.capturedRequest?.container).toBe('container_explicit');
  });

  test('sends no container when no message carries one', async () => {
    const model = new RecordingAnthropic();
    await run(model, [new HumanMessage('hello')]);

    expect(model.capturedRequest?.container).toBeUndefined();
  });
});

describe('getCurrentTurnContainerId', () => {
  test('reads the container from response_metadata', () => {
    const message = new AIMessage({
      content: 'x',
      response_metadata: { container: { id: 'container_meta' } },
    });

    expect(getCurrentTurnContainerId([new HumanMessage('go'), message])).toBe(
      'container_meta'
    );
  });

  test('keeps the id of a streamed chunk merge intact', () => {
    const start = new AIMessageChunk({
      content: '',
      additional_kwargs: {
        container: { id: 'container_011CZabc', expires_at: '2026-01-01T00:00' },
      },
    });
    const delta = new AIMessageChunk({
      content: '',
      additional_kwargs: {
        container: { id: 'container_011CZabc', expires_at: '2026-01-01T00:00' },
      },
    });

    expect(
      getCurrentTurnContainerId([new HumanMessage('go'), start.concat(delta)])
    ).toBe('container_011CZabc');
  });

  test('collapses an id that a merge repeated', () => {
    const message = aiWithContainer('container_011CZabccontainer_011CZabc');

    expect(getCurrentTurnContainerId([new HumanMessage('go'), message])).toBe(
      'container_011CZabc'
    );
  });

  test('ignores an AI message without a usable container id', () => {
    const messages = [
      new HumanMessage('go'),
      aiWithContainer('container_cur'),
      new AIMessage({
        content: 'x',
        additional_kwargs: { container: { id: '' } },
      }),
    ];

    expect(getCurrentTurnContainerId(messages)).toBe('container_cur');
  });

  test('treats an unmarked human message as the start of a turn', () => {
    const messages = [
      new HumanMessage('go'),
      aiWithContainer('container_cur'),
      new HumanMessage({
        content: 'meta without a known source',
        additional_kwargs: { isMeta: true },
      }),
    ];

    expect(getCurrentTurnContainerId(messages)).toBeUndefined();
  });
});

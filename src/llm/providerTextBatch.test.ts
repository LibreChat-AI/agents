import { createServer } from 'node:http';
import { MemorySaver } from '@langchain/langgraph';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type { ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
} from '@/protection/providerText';
import type { EventHandler, MessageDeltaEvent } from '@/types';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { withProviderTextBoundary } from '@/llm/providerTextBoundary';
import { createRemoteSdkRun } from '@/runtime/client';
import { createGraph } from '@/graphs/createGraph';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { attemptInvoke } from '@/llm/invoke';
import { ChatOpenAI } from '@/llm/openai';
import { Run } from '@/run';

const marker = 'private';
const allowed = (content: string): ProviderTextProtectionResult => ({
  version: 1,
  ok: true,
  value: { content, replacements: 0, categories: [] },
});
const blocked: ProviderTextProtectionResult = {
  version: 1,
  ok: false,
  error: { code: 'blocked' },
};
const inspect = ({
  content,
}: {
  content: string;
}): ProviderTextProtectionResult =>
  content.includes(marker) ? blocked : allowed(content);

function policy(
  overrides: Partial<ProviderTextProtection> = {}
): ProviderTextProtection {
  return {
    version: 1,
    timeoutMs: 5_000,
    maxAttemptBytes: 32_768,
    maxBufferedBytes: 131_072,
    batching: {
      version: 1,
      batchBytes: 16,
      flushMs: 1000,
      maxMatchChars: marker.length,
    },
    classify: () => 'prose',
    inspect,
    ...overrides,
  };
}

class Model extends FakeListChatModel {
  signal?: AbortSignal;
  reads = 0;
  constructor(
    private readonly frames: AsyncGenerator<string | AIMessageChunk>
  ) {
    super({ responses: ['unused'] });
  }
  override async *_streamResponseChunks(
    _messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): AsyncGenerator<ChatGenerationChunk> {
    this.signal = options.signal;
    try {
      while (options.signal?.aborted !== true) {
        this.reads++;
        const frame = await this.frames.next();
        if (frame.done === true) return;
        const message =
          typeof frame.value === 'string'
            ? new AIMessageChunk(frame.value)
            : frame.value;
        yield new ChatGenerationChunk({
          text: typeof message.content === 'string' ? message.content : '',
          message,
        });
      }
    } finally {
      await this.frames.return(undefined);
    }
  }
}

async function drain(
  model: Model,
  protection: ProviderTextProtection,
  seen: string[]
): Promise<void> {
  const guarded = withProviderTextBoundary(model, protection, Providers.OPENAI);
  const stream = await guarded.stream!([new HumanMessage('go')]);
  for await (const chunk of stream)
    if (typeof chunk.content === 'string' && chunk.content)
      seen.push(chunk.content);
}

it.each(Array.from({ length: marker.length - 1 }, (_, i) => i + 1))(
  'blocks a match split at character %i without committing its prefix',
  async (split) => {
    const seen: string[] = [];
    const model = new Model(
      (async function* () {
        yield 'allowed prefix ' + marker.slice(0, split);
        yield marker.slice(split) + ' after the boundary';
        await new Promise<void>((resolve) =>
          model.signal?.addEventListener('abort', () => resolve(), {
            once: true,
          })
        );
      })()
    );
    await expect(drain(model, policy(), seen)).rejects.toMatchObject({
      code: 'blocked',
    });
    expect(seen.join('')).toBe(
      'allowed prefix '.slice(
        0,
        'allowed prefix '.length - (marker.length - 1) + split
      )
    );
    expect(model.signal?.reason).toMatchObject({ code: 'blocked' });
  }
);

it('flushes during a provider stall and aborts when a later window matches', async () => {
  const release = Promise.withResolvers<void>();
  const readAhead = Promise.withResolvers<void>();
  const seen: string[] = [];
  const model = new Model(
    (async function* () {
      yield 'approved buffer ';
      readAhead.resolve();
      await release.promise;
      yield ' has private text';
      await new Promise<void>((resolve) =>
        model.signal?.addEventListener('abort', () => resolve(), { once: true })
      );
    })()
  );
  const protectedModel = withProviderTextBoundary(
    model,
    policy({
      batching: {
        version: 1,
        batchBytes: 128,
        flushMs: 10,
        maxMatchChars: marker.length,
      },
    }),
    Providers.OPENAI
  );
  const stream = await protectedModel.stream!([]);
  const collect = async (): Promise<void> => {
    for await (const chunk of stream) {
      if (typeof chunk.content !== 'string' || !chunk.content) continue;
      seen.push(chunk.content);
      await readAhead.promise;
      release.resolve();
    }
  };
  await expect(collect()).rejects.toMatchObject({ code: 'blocked' });
  expect(seen).toEqual(['approved b']);
  expect(model.signal?.reason).toMatchObject({ code: 'blocked' });
});

it('bounds producer read-ahead to one frame while the asynchronous inspector is held', async () => {
  const started = Promise.withResolvers<void>();
  const nextRead = Promise.withResolvers<void>();
  const verdict = Promise.withResolvers<void>();
  const seen: string[] = [];
  const model = new Model(
    (async function* () {
      yield 'first safe buffer ';
      nextRead.resolve();
      yield 'second safe buffer ';
      yield 'third safe buffer';
    })()
  );
  const inspected = jest.fn(async ({ content }: { content: string }) => {
    started.resolve();
    await verdict.promise;
    return allowed(content);
  });
  const result = drain(model, policy({ inspect: inspected }), seen);
  await Promise.all([started.promise, nextRead.promise]);
  expect(model.reads).toBe(2);
  expect(inspected).toHaveBeenCalledTimes(1);
  expect(seen).toEqual([]);
  verdict.resolve();
  await result;
  expect(seen.join('')).toBe(
    'first safe buffer second safe buffer third safe buffer'
  );
  expect(inspected).toHaveBeenCalledTimes(4);
});

it('keeps approved deltas on a blocked Run without committing an assistant message', async () => {
  const model = new Model(
    (async function* () {
      yield 'safe prefix here ';
      yield 'is private content';
      yield 'not released';
    })()
  );
  const handler = jest.fn();
  const run = await Run.create({
    runId: 'batched-block',
    skipCleanup: true,
    graphConfig: {
      type: 'standard',
      llmConfig: { provider: Providers.OPENAI },
    },
    customHandlers: { [GraphEvents.ON_MESSAGE_DELTA]: { handle: handler } },
    providerTextProtection: policy(),
  });
  run.Graph!.overrideModel = model;
  const parent = new AbortController();
  await expect(
    run.processStream(
      { messages: [new HumanMessage('go')] },
      { version: 'v2', signal: parent.signal }
    )
  ).rejects.toMatchObject({ code: 'blocked' });
  expect(parent.signal.aborted).toBe(false);
  expect(model.signal?.reason).toBeInstanceOf(ProviderTextProtectionError);
  expect(JSON.stringify(handler.mock.calls)).toContain('safe prefix');
  expect(JSON.stringify(handler.mock.calls)).not.toContain(marker);
  expect(
    run.Graph!.getRunMessages()?.some((message) => message._getType() === 'ai')
  ).not.toBe(true);
});

it('rejects reasoning without handing it to native observers', async () => {
  const model = new Model(
    (async function* () {
      yield new AIMessageChunk({
        content: '',
        additional_kwargs: { reasoning_content: marker },
      });
      yield 'safe';
    })()
  );
  const seen: string[] = [];
  await expect(drain(model, policy(), seen)).rejects.toMatchObject({
    code: 'unsupported',
  });
  expect(seen).toEqual([]);
});

it('recycles approved window leases across a longer response', async () => {
  const text = '0123456789 ordinary '.repeat(100);
  const model = new Model(
    (async function* () {
      for (let offset = 0; offset < text.length; offset += 20)
        yield text.slice(offset, offset + 20);
    })()
  );
  const seen: string[] = [];
  await drain(
    model,
    policy({ maxAttemptBytes: 2_048, maxBufferedBytes: 2_048 }),
    seen
  );
  expect(seen.join('')).toBe(text);
});

it('keeps shared capacity charged while a cancelled inspector drains', async () => {
  const entered = Promise.withResolvers<void>();
  const verdict = Promise.withResolvers<void>();
  const caller = new AbortController();
  const model = () =>
    new Model(
      (async function* () {
        yield 'safe '.repeat(50);
      })()
    );
  const protect = policy({
    maxAttemptBytes: 4_096,
    maxBufferedBytes: 4_096,
    async inspect({ content }) {
      entered.resolve();
      await verdict.promise;
      return allowed(content);
    },
  });
  const protectedModel = withProviderTextBoundary(
    model(),
    protect,
    Providers.OPENAI
  );
  const consumed: string[] = [];
  const result = (async () => {
    if (!(protectedModel instanceof BaseChatModel))
      throw new Error('Expected a chat model');
    for await (const chunk of protectedModel._streamResponseChunks([], {
      signal: caller.signal,
    })) {
      if (chunk.text) consumed.push(chunk.text);
    }
  })();
  const cancelled = expect(result).rejects.toMatchObject({ code: 'cancelled' });
  try {
    await entered.promise;
    caller.abort();
    await cancelled;
    await expect(drain(model(), protect, [])).rejects.toMatchObject({
      code: 'overflow',
    });
    expect(consumed).toEqual([]);
  } finally {
    verdict.resolve();
  }
  await new Promise<void>((resolve) => setImmediate(resolve));
  const next: string[] = [];
  await drain(model(), protect, next);
  expect(next.join('')).toBe('safe '.repeat(50));
});

it('does not publish a rewritten window', async () => {
  const model = new Model(
    (async function* () {
      yield 'first candidate';
    })()
  );
  const seen: string[] = [];
  await expect(
    drain(model, policy({ inspect: () => allowed('transformed') }), seen)
  ).rejects.toMatchObject({ code: 'incompatible' });
  expect(seen).toEqual([]);
});

it.each([false, true])(
  'uses the native OpenAI stream through completion or block (blocked=%s)',
  async (reject) => {
    const closed = Promise.withResolvers<void>();
    const prefix = 'An approved buffer. ';
    const suffix = reject ? 'Some private text.' : 'An approved ending.';
    let response: ServerResponse | undefined;
    const send = (
      delta: { role?: 'assistant'; content?: string },
      finish_reason: 'stop' | null = null,
      usage?: object
    ): void => {
      response?.write(
        `data: ${JSON.stringify({
          id: 'chatcmpl-fixture',
          object: 'chat.completion.chunk',
          created: 1,
          model: 'gpt-4o-mini',
          choices: usage == null ? [{ index: 0, delta, finish_reason }] : [],
          ...(usage && { usage }),
        })}\n\n`
      );
    };
    const server = createServer((req, res) => {
      req.resume();
      response = res;
      res.once('close', () => closed.resolve());
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      send({ role: 'assistant', content: prefix });
    });
    await new Promise<void>((resolve) =>
      server.listen(0, '127.0.0.1', resolve)
    );
    try {
      const { port } = server.address() as AddressInfo;
      const model = new ChatOpenAI({
        apiKey: 'fixture',
        model: 'gpt-4o-mini',
        streaming: true,
        streamUsage: true,
        useResponsesApi: false,
        configuration: { baseURL: `http://127.0.0.1:${port}/v1` },
      });
      let text = '';
      const task = attemptInvoke({
        model,
        provider: Providers.OPENAI,
        messages: [new HumanMessage('go')],
        providerTextProtection: policy(),
        onChunk: (chunk): void => {
          if (typeof chunk.content !== 'string' || chunk.content === '') return;
          const first = text === '';
          text += chunk.content;
          if (!first) return;
          send({ content: suffix });
          if (reject) return;
          send({}, 'stop');
          send({}, null, {
            prompt_tokens: 3,
            completion_tokens: 7,
            total_tokens: 10,
          });
          response?.end('data: [DONE]\n\n');
        },
      });
      if (reject) {
        await expect(task).rejects.toMatchObject({ code: 'blocked' });
        expect(text).toBe(prefix.slice(0, -(marker.length - 1)));
        await closed.promise;
      } else {
        const result = await task;
        expect(text).toBe(prefix + suffix);
        expect(result.messages?.at(-1)).toMatchObject({
          content: text,
          usage_metadata: {
            input_tokens: 3,
            output_tokens: 7,
            total_tokens: 10,
          },
        });
      }
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  }
);

it('keeps whole-reply protection when no bounded-window policy is supplied', async () => {
  const seen: string[] = [];
  const model = new Model(
    (async function* () {
      yield 'An otherwise approved beginning. ';
      yield 'With private text later.';
    })()
  );
  const inspectWhole = jest.fn(inspect);
  await expect(
    drain(model, policy({ batching: undefined, inspect: inspectWhole }), seen)
  ).rejects.toMatchObject({ code: 'blocked' });
  expect(inspectWhole).toHaveBeenCalledTimes(1);
  expect(seen).toEqual([]);
});

it('refuses a non-streaming invocation when the host requires batching', async () => {
  const model = new Model(
    (async function* () {
      yield 'Not read';
    })()
  );
  const guarded = withProviderTextBoundary(model, policy(), Providers.OPENAI);
  await expect(guarded.invoke([])).rejects.toMatchObject({
    code: 'unsupported',
  });
  expect(model.reads).toBe(0);
});

it('does not divide a UTF-16 surrogate pair at the commit boundary', async () => {
  const model = new Model(
    (async function* () {
      yield 'abcdefgh🧪';
      yield 'tail';
    })()
  );
  const seen: string[] = [];
  await drain(
    model,
    policy({
      batching: { version: 1, batchBytes: 8, flushMs: 100, maxMatchChars: 2 },
    }),
    seen
  );
  expect(seen[0]).toBe('abcdefgh');
  expect(seen.join('')).toBe('abcdefgh🧪tail');
});

it('rejects required batching before remote transport even with a whole-reply host service', async () => {
  const protect = policy();
  const host = jest.fn();
  await expect(
    createRemoteSdkRun({
      config: {
        runId: 'remote-batch',
        graphConfig: {
          type: 'standard',
          llmConfig: { provider: Providers.OPENAI },
        },
        providerTextProtection: protect,
      },
      identity: {
        runtimeId: 'worker',
        agentId: 'agent',
        userId: 'user',
        conversationId: 'thread',
        runId: 'remote-batch',
      },
      remote: {
        url: 'ws://127.0.0.1:9000',
        secret: 'test-only-fixture',
      },
      hostServices: {
        providerText: {
          descriptor: {
            version: 1,
            timeoutMs: protect.timeoutMs,
            maxAttemptBytes: protect.maxAttemptBytes,
            maxBufferedBytes: protect.maxBufferedBytes,
          },
          service: { version: 1, inspect: host },
        },
      },
    })
  ).rejects.toMatchObject({ code: 'unsupported' });
  expect(host).not.toHaveBeenCalled();
});

it('settles on timeout and releases raw capacity when an approved callback stalls', async () => {
  const stalled = Promise.withResolvers<void>();
  const started = Promise.withResolvers<void>();
  const protect = policy({
    timeoutMs: 120,
    maxAttemptBytes: 4_096,
    maxBufferedBytes: 4_096,
    batching: {
      version: 1,
      batchBytes: 16,
      flushMs: 10,
      maxMatchChars: marker.length,
    },
  });
  const frames = 'safe '.repeat(50);
  async function* producer(): AsyncGenerator<string> {
    yield frames;
    const signal = model.signal;
    if (signal?.aborted === false) {
      await new Promise<void>((resolve) =>
        signal.addEventListener('abort', () => resolve(), { once: true })
      );
    }
  }
  const model = new Model(producer());
  const run = attemptInvoke({
    model,
    provider: Providers.OPENAI,
    messages: [],
    providerTextProtection: protect,
    onChunk() {
      started.resolve();
      return stalled.promise;
    },
  });
  const result = expect(run).rejects.toMatchObject({ code: 'timeout' });
  try {
    await started.promise;
    await result;
    expect(model.signal?.reason).toMatchObject({ code: 'timeout' });
    const seen: string[] = [];
    await drain(
      new Model(
        (async function* () {
          yield frames;
        })()
      ),
      protect,
      seen
    );
    expect(seen.join('')).toBe(frames);
  } finally {
    stalled.resolve();
  }
});

it('rejects batching with a checkpointer when using the public graph factory', () => {
  const graph = createGraph({
    kind: 'standard',
    input: {
      agents: [{ agentId: 'text', provider: Providers.OPENAI }],
      providerTextProtection: policy(),
    },
  });
  graph.compileOptions = { checkpointer: new MemorySaver() };
  expect(() => graph.createWorkflow()).toThrow(
    expect.objectContaining({ code: 'unsupported' })
  );
});

it.each([false, true])(
  'awaits registered model deltas once under the attempt timeout (stall=%s)',
  async (stall) => {
    const text = 'An approved first batch. And the rest.';
    const stalled = Promise.withResolvers<void>();
    const unblockProvider = Promise.withResolvers<void>();
    const protect = policy({
      timeoutMs: 180,
      batching: {
        version: 1,
        batchBytes: 16,
        flushMs: 20,
        maxMatchChars: marker.length,
      },
    });
    const model = new Model(
      (async function* () {
        yield text.slice(0, 25);
        await unblockProvider.promise;
        yield text.slice(25);
      })()
    );
    const seen: string[] = [];
    const dispatch = new ChatModelStreamHandler();
    const handle = jest.fn<
      ReturnType<EventHandler['handle']>,
      Parameters<EventHandler['handle']>
    >((_event, data) => {
      const { delta } = data as MessageDeltaEvent;
      for (const part of delta.content ?? []) {
        if (part.type === 'text') seen.push(part.text);
      }
      if (stall) return stalled.promise;
      unblockProvider.resolve();
    });
    const run = await Run.create({
      runId: 'registered-batch',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
      },
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: dispatch,
        [GraphEvents.ON_MESSAGE_DELTA]: { handle },
      },
      providerTextProtection: protect,
    });
    run.Graph!.overrideModel = model;
    try {
      const result = run.processStream(
        { messages: [new HumanMessage('go')] },
        { version: 'v2' }
      );
      if (stall) {
        await expect(result).rejects.toMatchObject({ code: 'timeout' });
        expect(seen).toHaveLength(1);
        expect(model.signal?.reason).toMatchObject({ code: 'timeout' });
      } else {
        await result;
        expect(seen.join('')).toBe(text);
      }
    } finally {
      stalled.resolve();
      unblockProvider.resolve();
    }
  }
);

it.each(['reasoning', 'metadata'])(
  'rejects out-of-scope or oversized %s before staging a read-ahead frame',
  async (kind) => {
    const stalled = Promise.withResolvers<void>();
    const entered = Promise.withResolvers<void>();
    const protect = policy({
      maxAttemptBytes: 4_096,
      maxBufferedBytes: 4_096,
      async inspect({ content }) {
        entered.resolve();
        await stalled.promise;
        return allowed(content);
      },
    });
    const marker = 'x'.repeat(8192);
    const model = new Model(
      (async function* () {
        yield 'An approved first batch.';
        await entered.promise;
        yield new AIMessageChunk({
          content: '',
          ...(kind === 'reasoning'
            ? { additional_kwargs: { reasoning_content: marker } }
            : { response_metadata: { model_name: marker } }),
        });
      })()
    );
    const seen: string[] = [];
    try {
      await expect(drain(model, protect, seen)).rejects.toMatchObject({
        code: kind === 'reasoning' ? 'unsupported' : 'overflow',
      });
      expect(seen).toEqual([]);
      expect(model.signal?.aborted).toBe(true);
    } finally {
      stalled.resolve();
    }
  }
);

it('rejects batching on a standard Run with more than one configured agent', async () => {
  await expect(
    Run.create({
      runId: 'multi-entry-batch',
      graphConfig: {
        type: 'standard',
        agents: [
          { provider: Providers.OPENAI, agentId: 'first' },
          { provider: Providers.OPENAI, agentId: 'second' },
        ],
      },
      providerTextProtection: policy(),
    })
  ).rejects.toMatchObject({ code: 'unsupported' });
});

it.each([false, true])(
  'runs the registered start once before any batched delta (stall=%s)',
  async (stall) => {
    const entered = Promise.withResolvers<void>();
    const start = Promise.withResolvers<void>();
    const model = new Model(
      (async function* () {
        yield 'An approved first batch.';
      })()
    );
    const handleStart = jest.fn(async () => {
      entered.resolve();
      await start.promise;
    });
    const handleDelta = jest.fn();
    const run = await Run.create({
      runId: 'batched-start',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
      },
      customHandlers: {
        [GraphEvents.CHAT_MODEL_START]: { handle: handleStart },
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_MESSAGE_DELTA]: { handle: handleDelta },
      },
      providerTextProtection: policy({
        timeoutMs: 180,
        batching: {
          version: 1,
          batchBytes: 16,
          flushMs: 20,
          maxMatchChars: marker.length,
        },
      }),
    });
    run.Graph!.overrideModel = model;
    const task = run.processStream(
      { messages: [new HumanMessage('go')] },
      { version: 'v2' }
    );
    const result = stall
      ? expect(task).rejects.toMatchObject({ code: 'timeout' })
      : task;
    try {
      await entered.promise;
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(handleStart).toHaveBeenCalledTimes(1);
      expect(handleStart).toHaveBeenCalledWith(
        GraphEvents.CHAT_MODEL_START,
        { input: { messages: [[expect.objectContaining({ content: 'go' })]] } },
        expect.objectContaining({ provider: Providers.OPENAI }),
        run.Graph
      );
      expect(model.reads).toBe(0);
      expect(handleDelta).not.toHaveBeenCalled();
      if (!stall) start.resolve();
      await result;
      expect(handleStart).toHaveBeenCalledTimes(1);
      if (stall) expect(handleDelta).not.toHaveBeenCalled();
      else expect(handleDelta).toHaveBeenCalled();
    } finally {
      start.resolve();
    }
  }
);

it.each(['registered delta', 'native end'])(
  'does not retry or start a fallback when a %s callback rejects',
  async (kind) => {
    const fallback = jest.fn(() =>
      Promise.resolve(new Response('', { status: 500 }))
    );
    const text = 'An approved first batch.';
    const model = new Model(
      (async function* () {
        yield text;
      })()
    );
    const handle = jest.fn<
      ReturnType<EventHandler['handle']>,
      Parameters<EventHandler['handle']>
    >();
    const handleLLMEnd = jest.fn(() =>
      Promise.reject(new Error('host-only details'))
    );
    const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd });
    observer.raiseError = true;
    observer.awaitHandlers = true;
    if (kind === 'registered delta') {
      handle.mockRejectedValue(new Error('host-only details'));
    }
    const run = await Run.create({
      runId: 'batched-handler-error',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          fallbacks: [
            {
              provider: Providers.OPENAI,
              clientOptions: {
                apiKey: 'test-only-fixture',
                model: 'gpt-4o-mini',
                maxRetries: 0,
                configuration: { fetch: fallback },
              },
            },
          ],
        },
      },
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_MESSAGE_DELTA]: { handle },
      },
      providerTextProtection: policy(),
    });
    run.Graph!.overrideModel = model;
    const error: unknown = await run
      .processStream(
        { messages: [new HumanMessage('go')] },
        {
          version: 'v2',
          ...(kind === 'native end' && { callbacks: [observer] }),
        }
      )
      .catch((error: unknown) => error);
    expect(fallback).not.toHaveBeenCalled();
    expect(error).toMatchObject({
      code: 'unavailable',
      message: 'Provider text protection: unavailable',
    });
    expect(handle).toHaveBeenCalledTimes(kind === 'native end' ? 2 : 1);
    expect(handleLLMEnd).toHaveBeenCalledTimes(kind === 'native end' ? 1 : 0);
    if (kind === 'native end') {
      const visible = handle.mock.calls.flatMap(([, data]) => {
        const { delta } = data as MessageDeltaEvent;
        return (delta.content ?? []).map((part) =>
          part.type === 'text' ? part.text : ''
        );
      });
      expect(visible.join('')).toBe(text);
      expect(handle.mock.invocationCallOrder[0]).toBeLessThan(
        handleLLMEnd.mock.invocationCallOrder[0]
      );
    }
    expect(model.signal?.reason).toMatchObject({ code: 'unavailable' });
    expect(
      run
        .Graph!.getRunMessages()
        ?.some((message) => message._getType() === 'ai')
    ).not.toBe(true);
  }
);

it('still enforces the callback deadline after the provider reaches EOF', async () => {
  const stalled = Promise.withResolvers<void>();
  const eof = jest.fn();
  const model = new Model(
    (async function* () {
      yield 'Approved final text.';
      eof();
    })()
  );
  const onChunk = jest.fn(() => stalled.promise);
  try {
    await expect(
      attemptInvoke({
        model,
        provider: Providers.OPENAI,
        messages: [],
        providerTextProtection: policy({
          timeoutMs: 160,
          batching: {
            version: 1,
            batchBytes: 128,
            flushMs: 80,
            maxMatchChars: marker.length,
          },
        }),
        onChunk,
      })
    ).rejects.toMatchObject({ code: 'timeout' });
    expect(eof).toHaveBeenCalledTimes(1);
    expect(onChunk).toHaveBeenCalledTimes(1);
    expect(model.signal?.reason).toMatchObject({ code: 'timeout' });
  } finally {
    stalled.resolve();
  }
});

it('preserves a pre-publication block when a native error callback rejects', async () => {
  const fallback = jest.fn(() =>
    Promise.resolve(new Response('', { status: 500 }))
  );
  const handle = jest.fn();
  const handleLLMError = jest.fn(() =>
    Promise.reject(new Error('host-only details'))
  );
  const observer = BaseCallbackHandler.fromMethods({ handleLLMError });
  observer.raiseError = true;
  observer.awaitHandlers = true;
  const model = new Model(
    (async function* () {
      yield marker;
    })()
  );
  const run = await Run.create({
    runId: 'batched-error-handler',
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        fallbacks: [
          {
            provider: Providers.OPENAI,
            clientOptions: {
              apiKey: 'test-only-fixture',
              model: 'gpt-4o-mini',
              maxRetries: 0,
              configuration: { fetch: fallback },
            },
          },
        ],
      },
    },
    customHandlers: {
      [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
      [GraphEvents.ON_MESSAGE_DELTA]: { handle },
    },
    providerTextProtection: policy(),
  });
  run.Graph!.overrideModel = model;
  await expect(
    run.processStream(
      { messages: [new HumanMessage('go')] },
      { version: 'v2', callbacks: [observer] }
    )
  ).rejects.toMatchObject({ code: 'blocked' });
  expect(handleLLMError).toHaveBeenCalledTimes(1);
  expect(fallback).not.toHaveBeenCalled();
  expect(handle).not.toHaveBeenCalled();
  expect(model.signal?.reason).toMatchObject({ code: 'blocked' });
});

it.each([false, true])(
  'rejects late native completion (final delta stalled: %s)',
  async (stallDelta) => {
    const release = Promise.withResolvers<void>();
    const handleLLMEnd = jest.fn(() => release.promise);
    const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd });
    observer.awaitHandlers = true;
    const model = new Model(
      (async function* () {
        yield 'Approved final text.';
      })()
    );
    const onChunk = jest.fn(() => (stallDelta ? release.promise : undefined));
    const onModelComplete = jest.fn();
    // Settle the host callback only after the attempt is observed to time out.
    // The watchdog bounds the regression on implementations that hang instead.
    const watchdog = setTimeout(() => {
      watchdogFired = true;
      release.resolve();
    }, 600);
    let watchdogFired = false;
    try {
      await expect(
        attemptInvoke(
          {
            model,
            provider: Providers.OPENAI,
            messages: [],
            onChunk,
            onModelComplete,
            providerTextProtection: policy({
              timeoutMs: 120,
              batching: {
                version: 1,
                batchBytes: 128,
                flushMs: 60,
                maxMatchChars: marker.length,
              },
            }),
          },
          { callbacks: [observer] }
        )
      ).rejects.toMatchObject({ code: 'timeout' });
      expect(watchdogFired).toBe(false);
      expect(onChunk).toHaveBeenCalledTimes(1);
      expect(handleLLMEnd).toHaveBeenCalledTimes(1);
      expect(onModelComplete).not.toHaveBeenCalled();
      expect(model.signal?.reason).toMatchObject({ code: 'timeout' });
    } finally {
      clearTimeout(watchdog);
      release.resolve();
    }
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(onModelComplete).not.toHaveBeenCalled();
  }
);

it('times out a stalled native start without opening the provider', async () => {
  const start = Promise.withResolvers<void>();
  const handleChatModelStart = jest.fn(() => start.promise);
  const observer = BaseCallbackHandler.fromMethods({ handleChatModelStart });
  observer.awaitHandlers = true;
  const model = new Model(
    (async function* () {
      yield 'Approved text.';
    })()
  );
  const watchdog = setTimeout(start.resolve, 600);
  try {
    await expect(
      attemptInvoke(
        {
          model,
          provider: Providers.OPENAI,
          messages: [],
          providerTextProtection: policy({
            timeoutMs: 120,
            batching: {
              version: 1,
              batchBytes: 128,
              flushMs: 60,
              maxMatchChars: marker.length,
            },
          }),
        },
        { callbacks: [observer] }
      )
    ).rejects.toMatchObject({ code: 'timeout' });
    expect(handleChatModelStart).toHaveBeenCalledTimes(1);
  } finally {
    clearTimeout(watchdog);
    start.resolve();
  }
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(model.reads).toBe(0);
});

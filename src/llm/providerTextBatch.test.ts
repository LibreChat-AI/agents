import { createServer } from 'node:http';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type { ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
} from '@/protection/providerText';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { withProviderTextBoundary } from '@/llm/providerTextBoundary';
import { GraphEvents, Providers } from '@/common';
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

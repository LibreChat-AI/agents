import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
} from '@/protection/providerText';
import type { RuntimeProviderName, ChatModel, StreamEventData, EventHandler } from '@/types';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import fixtures from '@/protection/__tests__/fixtures/a1.json';
import { registerProvider } from '@/provider-registration';
import { ChatModelStreamHandler } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

function approved(content: string): ProviderTextProtectionResult {
  return {
    version: 1,
    ok: true,
    value: { content, replacements: 0, categories: [] },
  };
}

function policy(
  overrides: Partial<ProviderTextProtection> = {}
): ProviderTextProtection {
  return {
    version: 1,
    timeoutMs: 10_000,
    maxAttemptBytes: 64 * 1024,
    maxBufferedBytes: 256 * 1024,
    classify: () => 'prose',
    inspect: ({ content }) => approved(content),
    ...overrides,
  };
}

class Transport extends FakeListChatModel {
  calls = 0;
  readonly inputs: BaseMessage[][] = [];
  constructor(
    readonly chunks: AIMessageChunk[],
    readonly after?: () => Promise<void>
  ) {
    super({ responses: ['unused'] });
  }
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.calls++;
    this.inputs.push(messages);
    for (const message of this.chunks) {
      const chunk = new ChatGenerationChunk({
        text: typeof message.content === 'string' ? message.content : '',
        message,
      });
      await runManager?.handleLLMNewToken(
        chunk.text,
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk }
      );
      yield chunk;
    }
    await this.after?.();
  }
}

function invoke(
  model: ChatModel,
  protection?: ProviderTextProtection,
  callbacks?: BaseCallbackHandler[]
) {
  return attemptInvoke(
    {
      model,
      messages: [new HumanMessage('Allowed input')],
      provider: Providers.OPENAI,
      onChunk: async () => {},
      providerTextProtection: protection,
    },
    { callbacks }
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const assistantCases = fixtures.cases.filter(
  (entry) => entry.target.source === 'message' && entry.expected.content != null
);

describe('mandatory provider text boundary', () => {
  it.each(assistantCases)(
    'runs A1 $id through every two-chunk split',
    async (entry) => {
      const source = entry.chunks.join('');
      for (let split = 0; split <= source.length; split++) {
        const events: string[] = [];
        let inspected = false;
        const observer = BaseCallbackHandler.fromMethods({
          handleLLMNewToken: (token: string): void => {
            if (token) expect(inspected).toBe(true);
            events.push(token);
          },
          handleLLMEnd: (output): void => {
            events.push(JSON.stringify(output));
          },
        });
        observer.awaitHandlers = true;
        const result = await invoke(
          new Transport([
            new AIMessageChunk(source.slice(0, split)),
            new AIMessageChunk(source.slice(split)),
          ]),
          policy({
            inspect: ({ content }) => {
              expect(content).toBe(source);
              inspected = true;
              return approved(entry.expected.content!);
            },
          }),
          [observer]
        );
        expect(result.messages?.[0].content).toBe(entry.expected.content);
        for (const canary of fixtures.canaries)
          expect(events.join('')).not.toContain(canary);
        expect(events.join('')).toContain(entry.expected.content);
      }
    }
  );

  it('keeps absent-policy callback timing and text unchanged', async () => {
    const events: string[] = [];
    const observer = BaseCallbackHandler.fromMethods({
      handleLLMNewToken: (token): void => {
        events.push(token);
      },
    });
    observer.awaitHandlers = true;
    await invoke(
      new Transport([new AIMessageChunk(fixtures.canaries[0])]),
      undefined,
      [observer]
    );
    expect(events).toEqual([fixtures.canaries[0]]);
  });

  it('awaits the policy before native end callbacks, aggregate or onChunk release', async () => {
    const decision = deferred<ProviderTextProtectionResult>();
    const started = deferred<void>();
    const texts: string[] = [];
    const observer = BaseCallbackHandler.fromMethods({
      handleLLMEnd: (output): void => {
        texts.push(JSON.stringify(output));
      },
    });
    observer.awaitHandlers = true;
    const task = invoke(
      new Transport([new AIMessageChunk(fixtures.canaries[0])]),
      policy({
        inspect: () => {
          started.resolve();
          return decision.promise;
        },
      }),
      [observer]
    );
    await started.promise;
    expect(texts).toEqual([]);
    decision.resolve(approved('Approved control'));
    const result = await task;
    expect(result.messages?.[0].content).toBe('Approved control');
    expect(texts.join('')).toContain('Approved control');
    expect(texts.join('')).not.toContain(fixtures.canaries[0]);
  });

  it.each(['missing', 'throw', 'reject', 'version', 'block', 'overflow'])(
    'fails closed for %s',
    async (kind) => {
      const model = new Transport([new AIMessageChunk(fixtures.canaries[0])]);
      const protection = policy();
      if (kind === 'missing')
        Object.defineProperty(protection, 'inspect', { value: undefined });
      if (kind === 'version')
        Object.defineProperty(protection, 'version', { value: 2 });
      if (kind === 'overflow')
        Object.defineProperty(protection, 'maxAttemptBytes', { value: 512 });
      if (kind === 'throw')
        Object.defineProperty(protection, 'inspect', {
          value: () => {
            throw new Error(fixtures.canaries[0]);
          },
        });
      if (kind === 'reject')
        Object.defineProperty(protection, 'inspect', {
          value: () => Promise.reject(new Error(fixtures.canaries[0])),
        });
      if (kind === 'block')
        Object.defineProperty(protection, 'inspect', {
          value: () => ({ version: 1, ok: false, error: { code: 'blocked' } }),
        });
      const error = await invoke(model, protection).catch(
        (cause: Error) => cause
      );
      expect(error).toBeInstanceOf(ProviderTextProtectionError);
      expect(JSON.stringify(error)).not.toContain(fixtures.canaries[0]);
      if (kind === 'missing' || kind === 'version') expect(model.calls).toBe(0);
    }
  );

  it('quarantines Stop and late completion, then allows a fresh retry', async () => {
    const decision = deferred<ProviderTextProtectionResult>();
    const started = deferred<void>();
    const controller = new AbortController();
    const events: string[] = [];
    const protection = policy({
      inspect: () => {
        started.resolve();
        return decision.promise;
      },
    });
    const model = new Transport([new AIMessageChunk(fixtures.canaries[0])]);
    const task = attemptInvoke(
      {
        model,
        messages: [],
        provider: Providers.OPENAI,
        providerTextProtection: protection,
        onChunk: (chunk) => {
          events.push(JSON.stringify(chunk));
        },
      },
      { signal: controller.signal }
    );
    await started.promise;
    controller.abort();
    await expect(task).rejects.toMatchObject({ code: 'cancelled' });
    decision.resolve(approved('Late value'));
    await Promise.resolve();
    expect(events.join('')).not.toContain('Late value');
    expect(events.join('')).not.toContain(fixtures.canaries[0]);
    const retry = await invoke(
      model,
      policy({ inspect: () => approved('Retry control') })
    );
    expect(retry.messages?.[0].content).toBe('Retry control');
  });

  it('times out and retains a concurrent lease until an ignoring handler settles', async () => {
    const decision = deferred<ProviderTextProtectionResult>();
    const protection = policy({
      timeoutMs: 20,
      maxAttemptBytes: 4096,
      maxBufferedBytes: 4096,
      inspect: () => decision.promise,
    });
    const model = new Transport([new AIMessageChunk('x'.repeat(500))]);
    await expect(invoke(model, protection)).rejects.toMatchObject({
      code: 'timeout',
    });
    await expect(invoke(model, protection)).rejects.toMatchObject({
      code: 'overflow',
    });
    decision.resolve(approved('Control'));
    await new Promise((resolve) => setImmediate(resolve));
    expect((await invoke(model, protection)).messages?.[0].content).toBe(
      'Control'
    );
  });

  it('bounds parallel attempts together without mixing their text', async () => {
    const left = deferred<ProviderTextProtectionResult>();
    const right = deferred<ProviderTextProtectionResult>();
    const waiting = deferred<void>();
    const seen: string[] = [];
    const protection = policy({
      inspect: ({ content }) => {
        seen.push(content);
        if (seen.length === 2) waiting.resolve();
        return content === fixtures.canaries[0] ? left.promise : right.promise;
      },
    });
    const a = invoke(
      new Transport([new AIMessageChunk(fixtures.canaries[0])]),
      protection
    );
    const b = invoke(
      new Transport([new AIMessageChunk(fixtures.canaries[1])]),
      protection
    );
    await waiting.promise;
    left.resolve(approved('Tenant A control'));
    right.resolve(approved('Tenant B control'));
    expect((await a).messages?.[0].content).toBe('Tenant A control');
    expect((await b).messages?.[0].content).toBe('Tenant B control');
  });

  it('rejects code/JSON without rewriting it and rejects unsafe text aliases', async () => {
    const code = fixtures.cases
      .find((entry) => entry.id === 'code-inspect-only')!
      .chunks.join('');
    await expect(
      invoke(
        new Transport([new AIMessageChunk(code)]),
        policy({ classify: () => 'unsupported' })
      )
    ).rejects.toMatchObject({ code: 'unsupported' });
    await expect(
      invoke(
        new Transport([
          new AIMessageChunk({
            content: 'Control',
            additional_kwargs: { raw_text: fixtures.canaries[0] },
          }),
        ]),
        policy()
      )
    ).rejects.toMatchObject({ code: 'unsupported' });
  });

  it('preserves reasoning, signatures, tool args, IDs and usage before prose completes', async () => {
    const inspected: string[] = [];
    const chunk = new AIMessageChunk({
      id: 'provider-id',
      content: [
        {
          type: 'thinking',
          thinking: 'Reasoning control',
          signature: 'signed-control',
          index: 0,
        },
        { type: 'text', text: fixtures.canaries[0], index: 1 },
      ],
      tool_call_chunks: [
        { id: 'call-control', name: 'lookup', args: '{"count":42}', index: 0 },
      ],
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    });
    const result = await invoke(
      new Transport([chunk]),
      policy({
        inspect: ({ content }) => {
          inspected.push(content);
          return approved('Allowed prose');
        },
      })
    );
    expect(inspected).toEqual([fixtures.canaries[0]]);
    const message = result.messages?.[0] as AIMessageChunk;
    expect(message.id).toBe('provider-id');
    expect(message.tool_calls?.[0]).toMatchObject({
      id: 'call-control',
      name: 'lookup',
      args: { count: 42 },
    });
    expect(message.usage_metadata?.total_tokens).toBe(15);
    expect(JSON.stringify(message.content)).toContain('signed-control');
    expect(JSON.stringify(message.content)).toContain('Allowed prose');
    expect(JSON.stringify(message)).not.toContain(fixtures.canaries[0]);
  });

  it('covers invoke/disabled-streaming and rejects an uncertified custom runnable', async () => {
    const model = new Transport([]);
    model.disableStreaming = true;
    model.responses = [fixtures.canaries[0]];
    expect(
      (
        await invoke(
          model,
          policy({ inspect: () => approved('Invoke control') })
        )
      ).messages?.[0].content
    ).toBe('Invoke control');
    const custom: ChatModel = { invoke: async () => new AIMessageChunk('raw') };
    await expect(invoke(custom, policy())).rejects.toMatchObject({
      code: 'unsupported',
    });
  });

  it('does not enter fallback after a policy failure', async () => {
    await expect(
      tryFallbackProviders({
        fallbacks: [],
        messages: [],
        primaryError: new ProviderTextProtectionError('blocked'),
        providerTextProtection: policy(),
      })
    ).rejects.toMatchObject({ code: 'blocked' });
  });
});

it.each([false, true])(
  'protects real Run state and publication with registered dispatcher=%s',
  async (registered) => {
    const events: string[] = [];
    const protection = policy({
      inspect: ({ content }) => {
        expect(content).toBe(fixtures.canaries[0]);
        return approved('The result is 42.');
      },
    });
    const handlers: Record<string, EventHandler> = registered
      ? { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler() }
      : {};
    const run = await Run.create({
      runId: `provider-protection-${registered}`,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI, streaming: true },
        instructions: 'Be brief.',
      },
      providerTextProtection: protection,
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        ...handlers,
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (_event: string, data: StreamEventData): void => {
            events.push(JSON.stringify(data));
          },
        },
      },
    });
    run.Graph?.overrideTestModel([fixtures.canaries[0]], 1);
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      {
        configurable: { thread_id: `b1-${registered}` },
        version: 'v2',
      }
    );
    expect(JSON.stringify(run.Graph?.getRunMessages())).toContain(
      'The result is 42.'
    );
    expect(JSON.stringify(run.Graph?.getRunMessages())).not.toContain(
      fixtures.canaries[0]
    );
    expect(events.join('')).toContain('The result is 42.');
    expect(events.join('')).not.toContain(fixtures.canaries[0]);
  }
);

class FallbackTransport extends Transport {
  constructor(_config: object) { super([new AIMessageChunk('Fallback ' + fixtures.canaries[1])]); }
}

it('drops failed primary prefixes and protects actual fallback and retry attempts', async () => {
  const dispose = registerProvider({ provider: 'b1-fallback-test', model: FallbackTransport });
  try {
    const released: string[] = [];
    const observer = BaseCallbackHandler.fromMethods({ handleLLMNewToken: (token): void => { released.push(token); } });
    observer.awaitHandlers = true;
    const inspected: string[] = [];
    const protection = policy({ inspect: ({ content }) => { inspected.push(content); return approved('Fallback control'); } });
    const primary = new Transport([new AIMessageChunk(fixtures.canaries[0])], async () => { throw new Error('Transport unavailable'); });
    const primaryError = await invoke(primary, protection, [observer]).catch((error: Error) => error);
    expect(released.join('')).not.toContain(fixtures.canaries[0]);
    expect(inspected).toEqual([]);
    const fallback = await tryFallbackProviders({
      fallbacks: [{ provider: 'b1-fallback-test' as RuntimeProviderName, clientOptions: {} }],
      messages: [new HumanMessage('control')], primaryError, providerTextProtection: protection,
      onChunk: (chunk) => { released.push(JSON.stringify(chunk)); }, config: { callbacks: [observer] },
    });
    expect(fallback?.messages?.[0].content).toBe('Fallback control');
    expect(inspected).toEqual(['Fallback ' + fixtures.canaries[1]]);
    expect(released.join('')).toContain('Fallback control');
    for (const canary of fixtures.canaries) expect(released.join('')).not.toContain(canary);
    expect((await invoke(new FallbackTransport({}), protection)).messages?.[0].content).toBe('Fallback control');
  } finally { dispose(); }
});

it('gates the SDK OpenAI HTTP transport before token callbacks and native aggregation', async () => {
  const { ChatOpenAI } = await import('@/llm/openai');
  const text = fixtures.cases[0].chunks;
  const frames: { id: string; object: string; created: number; model: string; choices: { index: number; delta: { content: string; role?: string }; finish_reason: string | null }[] }[] = text.map((content, index) => ({
    id: 'chatcmpl-control', object: 'chat.completion.chunk', created: 1, model: 'synthetic-model',
    choices: [{ index: 0, delta: { content, ...(index === 0 ? { role: 'assistant' } : {}) }, finish_reason: null }],
  }));
  frames.push({ id: 'chatcmpl-control', object: 'chat.completion.chunk', created: 1, model: 'synthetic-model', choices: [{ index: 0, delta: { content: '' }, finish_reason: 'stop' }] });
  const model = new ChatOpenAI({
    model: 'synthetic-model', openAIApiKey: 'synthetic-test-key', streaming: true, streamUsage: false,
    lc_stream_delay: 0,
    configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } }) },
  });
  const observed: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({
    handleLLMNewToken: (token): void => { observed.push(token); },
    handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); },
  });
  observer.awaitHandlers = true;
  const result = await invoke(model, policy({ inspect: ({ content }) => {
    expect(content).toBe(text.join('')); return approved(fixtures.cases[0].expected.content!);
  } }), [observer]);
  expect(result.messages?.[0].content).toBe(fixtures.cases[0].expected.content);
  expect(observed.join('')).toContain('[EMAIL_1]');
  for (const canary of fixtures.canaries) expect(observed.join('')).not.toContain(canary);
});

class ReuseTransport extends FakeChatModel {
  readonly inputs: BaseMessage[][] = [];
  override async *_streamResponseChunks(messages: BaseMessage[], options: this['ParsedCallOptions'], runManager?: CallbackManagerForLLMRun): AsyncGenerator<ChatGenerationChunk> {
    this.inputs.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

it('uses only canonical prose in the next real model/tool turn', async () => {
  let executed = 0;
  const lookup = new DynamicStructuredTool({ name: 'lookup', description: 'Returns a control.', schema: z.object({ count: z.number() }), func: async ({ count }) => { expect(count).toBe(42); executed++; return 'Tool control'; } });
  const model = new ReuseTransport({ responses: [fixtures.canaries[0], 'Final control'], splitStrategy: { type: 'fixed', value: 1 }, toolCalls: [{ id: 'reuse-call', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  const run = await Run.create({ runId: 'b1-reuse', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI, streaming: true }, tools: [lookup] }, providerTextProtection: policy({ inspect: ({ content }) => approved(content === fixtures.canaries[0] ? '[EMAIL_1]' : content) }), returnContent: true, skipCleanup: true });
  run.Graph!.overrideModel = model;
  await run.processStream({ messages: [new HumanMessage('control')] }, { version: 'v2', configurable: { thread_id: 'b1-reuse' } });
  expect(executed).toBe(1);
  expect(model.inputs).toHaveLength(2);
  const reused = JSON.stringify(model.inputs[1]);
  expect(reused).toContain('[EMAIL_1]');
  expect(reused).toContain('reuse-call');
  expect(reused).toContain('Tool control');
  expect(reused).not.toContain(fixtures.canaries[0]);
});

it('keeps real Run Stop state and message events free of undecided and late prose', async () => {
  const started = deferred<void>();
  const decision = deferred<ProviderTextProtectionResult>();
  const controller = new AbortController();
  const observed: string[] = [];
  const run = await Run.create({ runId: 'b1-stop', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI, streaming: true } }, providerTextProtection: policy({ inspect: () => { started.resolve(); return decision.promise; } }), skipCleanup: true, customHandlers: { [GraphEvents.ON_MESSAGE_DELTA]: { handle: (_event, data): void => { observed.push(JSON.stringify(data)); } } } });
  run.Graph!.overrideTestModel([fixtures.canaries[0]], 1);
  const task = run.processStream({ messages: [new HumanMessage('control')] }, { signal: controller.signal, version: 'v2', configurable: { thread_id: 'b1-stop' } });
  await started.promise;
  controller.abort();
  await task.catch(() => {});
  decision.resolve(approved('Late control'));
  await new Promise((resolve) => setImmediate(resolve));
  const payload = observed.join('') + JSON.stringify(run.Graph?.getRunMessages());
  expect(payload).not.toContain(fixtures.canaries[0]);
  expect(payload).not.toContain('Late control');
});

it('rejects raw aliases in response metadata and native non-streaming results', async () => {
  await expect(invoke(new Transport([new AIMessageChunk({ content: 'Control', response_metadata: { raw_text: fixtures.canaries[0] } })]), policy())).rejects.toMatchObject({ code: 'unsupported' });
  const model = new Transport([]);
  model.disableStreaming = true;
  model._generate = async () => ({ generations: [{ text: 'Control', message: new AIMessageChunk('Control') }], llmOutput: { raw_text: fixtures.canaries[0] } });
  await expect(invoke(model, policy())).rejects.toMatchObject({ code: 'unsupported' });
});

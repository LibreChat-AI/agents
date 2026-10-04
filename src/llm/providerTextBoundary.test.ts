import { z } from 'zod';
import { getEventListeners } from 'node:events';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { RunnableBinding, RunnableLambda } from '@langchain/core/runnables';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { ConverseStreamCommandOutput, ConverseStreamOutput } from '@aws-sdk/client-bedrock-runtime';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
} from '@/protection/providerText';
import type { RuntimeProviderName, ChatModel, StreamEventData, EventHandler } from '@/types';
import { ProviderTextAttempt, ProviderTextProtectionError } from '@/protection/providerText';
import { withProviderTextBoundary } from '@/llm/providerTextBoundary';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import fixtures from '@/protection/__tests__/fixtures/a1.json';
import { registerProvider } from '@/provider-registration';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { AgentContext } from '@/agents/AgentContext';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatModelStreamHandler } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { ChatOpenAI } from '@/llm/openai';
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

it('rejects a BaseChatModel subclass that bypasses the certified stream iterator', async () => {
  const model = new Transport([new AIMessageChunk('Control')]);
  Object.defineProperty(model, '_streamIterator', { value: async function* () { yield new AIMessageChunk(fixtures.canaries[0]); } });
  await expect(invoke(model, policy())).rejects.toMatchObject({ code: 'unsupported' });
});

it('rejects a required policy placed in a legacy graph config instead of RunConfig', async () => {
  const graphConfig = { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, providerTextProtection: policy() };
  await expect(Run.create({ runId: 'misplaced-policy', graphConfig })).rejects.toMatchObject({ code: 'incompatible' });
});

it('preserves real Anthropic lifecycle controls and usage while protecting prose', async () => {
  const text = fixtures.cases[0].chunks;
  const frames = [
    { type: 'message_start', message: { id: 'msg-control', type: 'message', role: 'assistant', model: 'claude-test', content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 10, output_tokens: 0, cache_read_input_tokens: 2, cache_creation_input_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    ...text.map((chunk) => ({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: chunk } })),
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 5 } },
    { type: 'message_stop' },
  ];
  const model = new CustomAnthropic({ anthropicApiKey: 'synthetic-test-key', model: 'claude-test', streaming: true, _lc_stream_delay: 0, clientOptions: { fetch: async () => new Response(frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } }) } });
  const observed: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); } });
  observer.awaitHandlers = true;
  const result = await attemptInvoke({ model, messages: [new HumanMessage('control')], provider: Providers.ANTHROPIC, onChunk: (chunk) => { observed.push(JSON.stringify(chunk)); }, providerTextProtection: policy({ inspect: ({ content }) => { expect(content).toBe(text.join('')); return approved('Anthropic control'); } }) }, { callbacks: [observer] });
  const message = result.messages?.[0] as AIMessageChunk;
  expect(message.content).toBe('Anthropic control');
  expect(message.id).toBe('msg-control');
  expect(message.additional_kwargs).toMatchObject({ id: 'msg-control', type: 'message', role: 'assistant', model: 'claude-test', stop_reason: 'end_turn' });
  expect(message.usage_metadata).toMatchObject({ input_tokens: 12, output_tokens: 5 });
  expect(observed.join('')).toContain('Anthropic control');
  for (const canary of fixtures.canaries) expect(observed.join('')).not.toContain(canary);
});

it('preserves real Bedrock block indices, tool seals, lifecycle controls and usage', async () => {
  const model = new CustomChatBedrockConverse({ model: 'anthropic.claude-test', region: 'us-east-1', credentials: { accessKeyId: 'synthetic', secretAccessKey: 'synthetic' }, _lc_stream_delay: 0 });
  const frames: ConverseStreamOutput[] = [
    { messageStart: { role: 'assistant' } },
    ...fixtures.cases[0].chunks.map((text) => ({ contentBlockDelta: { contentBlockIndex: 0, delta: { text } } })),
    { contentBlockStart: { contentBlockIndex: 1, start: { toolUse: { toolUseId: 'bedrock-call', name: 'lookup' } } } },
    { contentBlockDelta: { contentBlockIndex: 1, delta: { toolUse: { input: '{"count":42}' } } } },
    { contentBlockStop: { contentBlockIndex: 1 } },
    { messageStop: { stopReason: 'tool_use' } },
    { metadata: { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 }, metrics: { latencyMs: 3 } } },
  ];
  const send = jest.spyOn(model.client, 'send').mockImplementation(async (): Promise<ConverseStreamCommandOutput> => ({ $metadata: {}, stream: (async function* () { yield* frames; })() }));
  try {
    const observed: string[] = [];
    const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); } });
    observer.awaitHandlers = true;
    const result = await attemptInvoke({ model, messages: [new HumanMessage('control')], provider: Providers.BEDROCK, onChunk: (chunk) => { observed.push(JSON.stringify(chunk)); }, providerTextProtection: policy({ inspect: () => approved('Bedrock control') }) }, { callbacks: [observer] });
    const message = result.messages?.[0] as AIMessageChunk;
    expect(message.content).toBe('Bedrock control');
    expect(message.tool_calls).toEqual([{ id: 'bedrock-call', name: 'lookup', args: { count: 42 }, type: 'tool_call' }]);
    const baseline = await attemptInvoke({ model, messages: [new HumanMessage('control')], provider: Providers.BEDROCK, onChunk: async () => {} });
    expect(message.response_metadata).toEqual(baseline.messages?.[0].response_metadata);
    expect(message.response_metadata).toMatchObject({ lc_streamed_tool_call_seal: { kind: 'single', index: 1 }, messageStart: { role: 'assistant' }, messageStop: { stopReason: 'tool_use' } });
    expect(observed.join('')).toContain('bedrock_converse');
    expect(message.usage_metadata?.total_tokens).toBe(15);
    expect(observed.join('')).toContain('Bedrock control');
    for (const canary of fixtures.canaries) expect(observed.join('')).not.toContain(canary);
  } finally { send.mockRestore(); }
});

it('preserves official OpenAI tool-delta adapter controls alongside protected prose', async () => {
  const frames = [
    ...fixtures.cases[0].chunks.map((content, index) => ({ choices: [{ index: 0, delta: { content, ...(index === 0 ? { role: 'assistant' } : {}) }, finish_reason: null }] })),
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'openai-call', type: 'function', function: { name: 'lookup', arguments: '{"count":' } }] }, finish_reason: null }] },
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '42}' } }] }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
  ].map((frame) => ({ ...frame, id: 'chatcmpl-control', object: 'chat.completion.chunk', created: 1, model: 'synthetic-model' }));
  const model = new ChatOpenAI({ model: 'synthetic-model', apiKey: 'synthetic-test-key', firstPartyEndpoint: true, streaming: true, streamUsage: false, _lc_stream_delay: 0, configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } }) } });
  const observed: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); } });
  observer.awaitHandlers = true;
  const result = await invoke(model, policy({ inspect: () => approved('OpenAI control') }), [observer]);
  const message = result.messages?.[0] as AIMessageChunk;
  expect(message.tool_calls).toEqual([{ id: 'openai-call', name: 'lookup', args: { count: 42 }, type: 'tool_call' }]);
  const baseline = await invoke(model);
  expect(message.response_metadata).toEqual(baseline.messages?.[0].response_metadata);
  expect(observed.join('')).toContain('openai_chat_sequential');
  expect(message.content).toBe('OpenAI control');
  expect(observed.join('')).toContain('OpenAI control');
  for (const canary of fixtures.canaries) expect(observed.join('')).not.toContain(canary);
});

it('removes every completed wait listener instead of retaining losing cancellation races', async () => {
  const attempt = new ProviderTextAttempt(policy({ maxAttemptBytes: 512, maxBufferedBytes: 512 }));
  try {
    for (let index = 0; index < 30_000; index++) {
      await attempt.wait(Promise.resolve({ control: index }));
      expect(getEventListeners(attempt.signal, 'abort')).toHaveLength(0);
    }
  } finally { attempt.finish(); }
});

it('charges empty and control chunks against the attempt budget', async () => {
  const model = new Transport(Array.from({ length: 20 }, () => new AIMessageChunk('')));
  await expect(invoke(model, policy({ maxAttemptBytes: 1024 }))).rejects.toMatchObject({ code: 'overflow' });
});

it.each([
  { contentBlockIndex: 'unchecked alias' },
  { lc_streamed_tool_call_adapter: 'unknown' },
  { lc_streamed_tool_call_seal: { kind: 'all', raw: fixtures.canaries[0] } },
  { messageStart: { role: 'assistant', raw: fixtures.canaries[0] } },
  { metadata: { usage: { raw: fixtures.canaries[0] } } },
])('rejects malformed or unchecked provider controls %#', async (response_metadata) => {
  await expect(invoke(new Transport([new AIMessageChunk({ content: 'Control', response_metadata })]), policy())).rejects.toMatchObject({ code: 'unsupported' });
});

it('keeps an ignoring producer charged and quarantines its late end after cancellation', async () => {
  const draining = deferred<void>();
  const waiting = deferred<void>();
  const controller = new AbortController();
  const seen: string[] = [];
  const protection = policy({ maxAttemptBytes: 1024, maxBufferedBytes: 1024, inspect: ({ content }) => { seen.push(content); return approved('Control'); } });
  const model = new Transport([new AIMessageChunk(fixtures.canaries[0])], () => { waiting.resolve(); return draining.promise; });
  const task = attemptInvoke({ model, messages: [], provider: Providers.OPENAI, onChunk: async () => {}, providerTextProtection: protection }, { signal: controller.signal });
  await waiting.promise;
  controller.abort();
  await expect(task).rejects.toMatchObject({ code: 'cancelled' });
  await expect(invoke(new Transport([new AIMessageChunk('control')]), protection)).rejects.toMatchObject({ code: 'overflow' });
  draining.resolve();
  await new Promise((resolve) => setImmediate(resolve));
  expect(seen).toEqual([]);
  expect((await invoke(new Transport([new AIMessageChunk('control')]), protection)).messages?.[0].content).toBe('Control');
});

it('releases a failed synchronous transport creation lease before a fresh attempt', async () => {
  const model = new Transport([]);
  model._streamResponseChunks = () => { throw new Error('Transport creation failed'); };
  const protection = policy({ maxAttemptBytes: 1024, maxBufferedBytes: 1024 });
  await expect(invoke(model, protection)).rejects.toThrow('Transport creation failed');
  expect((await invoke(new Transport([new AIMessageChunk('Control')]), protection)).messages?.[0].content).toBe('Control');
});

it('rejects a multi-provider sequence before either producer or native callback starts', async () => {
  const first = new Transport([new AIMessageChunk(fixtures.canaries[0])]);
  const second = new Transport([new AIMessageChunk('Final control')]);
  const observed: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({ handleLLMNewToken: (token): void => { observed.push(token); } });
  observer.awaitHandlers = true;
  const sequence = first.pipe(new StringOutputParser()).pipe(second);
  await expect(invoke(sequence, policy(), [observer])).rejects.toMatchObject({ code: 'unsupported' });
  expect(first.calls).toBe(0);
  expect(second.calls).toBe(0);
  expect(observed).toEqual([]);
});

it('rejects opaque prefix callbacks that could run an unchecked producer', async () => {
  let started = false;
  const prefix = RunnableLambda.from(async () => { started = true; return 'unchecked'; });
  await expect(invoke(prefix.pipe(new Transport([new AIMessageChunk('Control')])), policy())).rejects.toMatchObject({ code: 'unsupported' });
  expect(started).toBe(false);
});

it.each(['anthropic', 'openai'])('rejects %s internally streaming invoke before producer allocation', async (provider) => {
  let requests = 0;
  const fetch = async (): Promise<Response> => { requests++; throw new Error('Producer must not start'); };
  const model = provider === 'anthropic'
    ? new CustomAnthropic({ anthropicApiKey: 'synthetic-test-key', model: 'claude-test', disableStreaming: true, streaming: true, maxRetries: 0, clientOptions: { fetch } })
    : new ChatOpenAI({ apiKey: 'synthetic-test-key', model: 'synthetic-model', disableStreaming: true, streaming: true, maxRetries: 0, configuration: { apiKey: 'synthetic-test-key', fetch } });
  model.disableStreaming = true;
  model.streaming = true;
  const protection = policy({ maxAttemptBytes: 512, maxBufferedBytes: 512 });
  await expect(invoke(model, protection)).rejects.toMatchObject({ code: 'unsupported' });
  expect(requests).toBe(0);
  const attempts = await Promise.allSettled([invoke(model, protection), invoke(model, protection)]);
  expect(attempts.map((attempt) => attempt.status)).toEqual(['rejected', 'rejected']);
  expect(requests).toBe(0);
});

it('preserves validated native Anthropic nonstreaming lifecycle metadata before release', async () => {
  const response = { id: 'msg-nonstream-control', type: 'message', role: 'assistant', model: 'claude-test', content: [{ type: 'text', text: fixtures.canaries[0] }], stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: 10, output_tokens: 5 } };
  const model = new CustomAnthropic({ anthropicApiKey: 'synthetic-test-key', model: 'claude-test', disableStreaming: true, streaming: false, clientOptions: { fetch: async () => new Response(JSON.stringify(response), { headers: { 'content-type': 'application/json' } }) } });
  const observed: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); } });
  observer.awaitHandlers = true;
  model.disableStreaming = true;
  const result = await invoke(model, policy({ inspect: ({ content }) => { expect(content).toBe(fixtures.canaries[0]); return approved('Native control'); } }), [observer]);
  const message = result.messages?.[0] as AIMessageChunk;
  expect(message.content).toBe('Native control');
  expect(message.id).toBe('msg-nonstream-control');
  expect(message.response_metadata).toMatchObject({ id: 'msg-nonstream-control', model: 'claude-test', stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: 10, output_tokens: 5 } });
  expect(message.usage_metadata).toMatchObject({ input_tokens: 10, output_tokens: 5, total_tokens: 15 });
  expect(observed.join('')).toContain('Native control');
  for (const canary of fixtures.canaries) expect(observed.join('')).not.toContain(canary);
});

it('rejects OpenAI internal streaming delegates even when the facade streaming flag is false', async () => {
  let requests = 0;
  const model = new ChatOpenAI({ apiKey: 'synthetic-test-key', model: 'synthetic-model', streaming: false, maxRetries: 0, configuration: { apiKey: 'synthetic-test-key', fetch: async () => { requests++; throw new Error('Producer must not start'); } } });
  model.disableStreaming = true;
  const delegate: unknown = Reflect.get(model, 'completions');
  expect(delegate).toBeInstanceOf(BaseChatModel);
  Object.defineProperty(delegate, 'streaming', { value: true });
  await expect(invoke(model, policy())).rejects.toMatchObject({ code: 'unsupported' });
  expect(requests).toBe(0);
});

it('rejects invoke selected without a streaming callback before internal aggregation', async () => {
  let requests = 0;
  const model = new CustomAnthropic({ anthropicApiKey: 'synthetic-test-key', model: 'claude-test', streaming: true, maxRetries: 0, clientOptions: { fetch: async () => { requests++; throw new Error('Producer must not start'); } } });
  const protectedModel = withProviderTextBoundary(model, policy());
  await expect(protectedModel.invoke([new HumanMessage('control')])).rejects.toMatchObject({ code: 'unsupported' });
  expect(requests).toBe(0);
});

it('rejects mutated bindings and callback config factories before any producer starts', async () => {
  const model = new Transport([new AIMessageChunk('Control')]);
  const bound = model.withConfig({ runName: 'bound' });
  Object.defineProperty(bound, 'stream', { value: model.stream.bind(model) });
  await expect(invoke(bound, policy())).rejects.toMatchObject({ code: 'unsupported' });
  let factoryStarted = false;
  const factory = model.withListeners({ onStart: () => { factoryStarted = true; } });
  await expect(invoke(factory, policy())).rejects.toMatchObject({ code: 'unsupported' });
  expect(model.calls).toBe(0);
  expect(factoryStarted).toBe(false);
});

it.each(['function', 'prototype'])('rejects a modified SDK instruction prefix (%s)', async (mutation) => {
  const context = AgentContext.fromConfig({ agentId: 'control', provider: Providers.OPENAI, instructions: 'Allowed instructions.' });
  const prefix = context.systemRunnable;
  if (!(prefix instanceof RunnableBinding)) throw new Error('Expected SDK instruction binding');
  if (mutation === 'function') Object.defineProperty(prefix.bound, 'func', { value: async () => 'Unchecked prefix' });
  else Object.setPrototypeOf(prefix.bound, Object.create(Object.getPrototypeOf(prefix.bound)));
  const model = new Transport([new AIMessageChunk('Control')]);
  await expect(invoke(prefix.pipe(model), policy())).rejects.toMatchObject({ code: 'unsupported' });
  expect(model.calls).toBe(0);
});

it('rejects a terminal provider custom transform before Run publication or state', async () => {
  const model = new Transport([new AIMessageChunk(fixtures.canaries[0])]);
  let transformed = false;
  model.transform = async function* () {
    transformed = true;
    yield new AIMessageChunk(fixtures.canaries[0]);
  };
  let inspected = false;
  const events: string[] = [];
  const run = await Run.create({
    runId: 'b1-transform-admission',
    graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, instructions: 'Allowed prefix.' },
    providerTextProtection: policy({ inspect: () => { inspected = true; return { version: 1, ok: false, error: { code: 'blocked' } }; } }),
    skipCleanup: true,
    customHandlers: { [GraphEvents.ON_MESSAGE_DELTA]: { handle: (_event, data): void => { events.push(JSON.stringify(data)); } } },
  });
  run.Graph!.overrideModel = model;
  await expect(run.processStream({ messages: [new HumanMessage('control')] }, { version: 'v2', configurable: { thread_id: 'b1-transform-admission' } })).rejects.toMatchObject({ code: 'unsupported' });
  expect(transformed).toBe(false);
  expect(model.calls).toBe(0);
  expect(inspected).toBe(false);
  expect(events.join('')).not.toContain(fixtures.canaries[0]);
  expect(JSON.stringify(run.Graph?.getRunMessages())).not.toContain(fixtures.canaries[0]);
});

it.each([true, 'truthy'])('rejects OpenAI effective internal streaming from modelKwargs (%s)', async (stream) => {
  let requests = 0;
  const model = new ChatOpenAI({
    apiKey: 'synthetic-test-key', model: 'synthetic-model', streaming: false, modelKwargs: { stream }, maxRetries: 0,
    configuration: { apiKey: 'synthetic-test-key', fetch: async () => { requests++; throw new Error('Producer must not start'); } },
  });
  model.disableStreaming = true;
  const protection = policy({ maxAttemptBytes: 512, maxBufferedBytes: 512 });
  await expect(invoke(model, protection)).rejects.toMatchObject({ code: 'unsupported' });
  expect(requests).toBe(0);
  const attempts = await Promise.allSettled([invoke(model, protection), invoke(model, protection)]);
  expect(attempts.map((attempt) => attempt.status)).toEqual(['rejected', 'rejected']);
  expect(requests).toBe(0);
});

it('checks the effective per-invocation parameters before original generation', async () => {
  const model = new Transport([]);
  let generated = false;
  model._generate = async () => { generated = true; return { generations: [{ text: 'unchecked', message: new AIMessageChunk('unchecked') }] }; };
  model.invocationParams = (options) => ({ stream: options?.stop?.[0] === 'internal-stream' });
  model.disableStreaming = true;
  const protectedModel = withProviderTextBoundary(model, policy());
  if (!(protectedModel instanceof BaseChatModel)) throw new Error('Expected protected chat model');
  await expect(protectedModel.invoke([new HumanMessage('control')], { stop: ['internal-stream'] })).rejects.toMatchObject({ code: 'unsupported' });
  expect(generated).toBe(false);
});

it('still permits effective streaming through the guarded provider iterator', async () => {
  const frames = fixtures.cases[0].chunks.map((content, index) => ({ id: 'safe-stream-control', object: 'chat.completion.chunk', created: 1, model: 'synthetic-model', choices: [{ index: 0, delta: { content, ...(index === 0 ? { role: 'assistant' } : {}) }, finish_reason: null }] }));
  const model = new ChatOpenAI({ apiKey: 'synthetic-test-key', model: 'synthetic-model', streaming: false, modelKwargs: { stream: true }, streamUsage: false, _lc_stream_delay: 0, configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } }) } });
  model.disableStreaming = false;
  const result = await invoke(model, policy({ inspect: ({ content }) => { expect(content).toBe(fixtures.cases[0].chunks.join('')); return approved('Guarded control'); } }));
  expect(result.messages?.[0].content).toBe('Guarded control');
  expect(JSON.stringify(result)).not.toContain(fixtures.canaries[0]);
});

it.each([false, true])('preserves real Anthropic tool-input fragments in Run dispatch (registered=%s)', async (registered) => {
  const executed: number[] = [];
  const lookup = new DynamicStructuredTool({ name: 'lookup', description: 'Returns a control.', schema: z.object({ count: z.number() }), func: async ({ count }) => { executed.push(count); return `Tool control ${count}`; } });
  const frames = [
    { type: 'message_start', message: { id: 'anthropic-tool-control', type: 'message', role: 'assistant', model: 'claude-test', content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 10, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    ...fixtures.cases[0].chunks.map((text) => ({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } })),
    { type: 'content_block_stop', index: 0 },
    { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'anthropic-call-1', name: 'lookup', input: {} } },
    { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"count":' } },
    { type: 'content_block_start', index: 2, content_block: { type: 'tool_use', id: 'anthropic-call-2', name: 'lookup', input: {} } },
    { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{"count":43}' } },
    { type: 'content_block_stop', index: 2 },
    { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '42}' } },
    { type: 'content_block_stop', index: 1 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 5 } },
    { type: 'message_stop' },
  ];
  const finalFrames = [
    { type: 'message_start', message: { id: 'anthropic-final-control', type: 'message', role: 'assistant', model: 'claude-test', content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 10, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Final control' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 5 } },
    { type: 'message_stop' },
  ];
  let requests = 0;
  const model = new CustomAnthropic({ anthropicApiKey: 'synthetic-test-key', model: 'claude-test', streaming: true, _lc_stream_delay: 0, clientOptions: { fetch: async () => {
    const responseFrames = requests++ === 0 ? frames : finalFrames;
    return new Response(responseFrames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } });
  } } });
  const inspected: string[] = [];
  const events: string[] = [];
  const handlers: Record<string, EventHandler> = registered ? { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler() } : {};
  const run = await Run.create({ runId: `anthropic-tools-${registered}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.ANTHROPIC, streaming: true }, instructions: 'Allowed instructions.', tools: [lookup] }, providerTextProtection: policy({ inspect: ({ content }) => { inspected.push(content); return approved(content === fixtures.cases[0].chunks.join('') ? fixtures.cases[0].expected.content! : content); } }), returnContent: true, skipCleanup: true, customHandlers: { ...handlers, [GraphEvents.ON_MESSAGE_DELTA]: { handle: (_event, data): void => { events.push(JSON.stringify(data)); } } } });
  run.Graph!.overrideModel = model.bindTools([lookup]);
  await run.processStream({ messages: [new HumanMessage('control')] }, { version: 'v2', configurable: { thread_id: `anthropic-tools-${registered}` } });
  expect(executed.sort()).toEqual([42, 43]);
  expect(requests).toBe(2);
  expect(inspected).toEqual([fixtures.cases[0].chunks.join(''), 'Final control']);
  const messages = run.Graph!.getRunMessages()!;
  const toolTurn = messages.find((message) => message.getType() === 'ai' && JSON.stringify(message).includes('anthropic-call-1')) as AIMessageChunk;
  expect(toolTurn.tool_calls).toEqual(expect.arrayContaining([{ id: 'anthropic-call-1', name: 'lookup', args: { count: 42 }, type: 'tool_call' }, { id: 'anthropic-call-2', name: 'lookup', args: { count: 43 }, type: 'tool_call' }]));
  expect(JSON.stringify(toolTurn.content)).toContain('tool_use');
  expect(events.join('')).toContain('Contact [EMAIL_1].');
  expect(JSON.stringify(messages)).toContain('Final control');
  for (const canary of fixtures.canaries) { expect(events.join('')).not.toContain(canary); expect(JSON.stringify(messages)).not.toContain(canary); }
});

it.each(['classify', 'inspect'])('fails closed when synchronous %s work outlives its deadline', async (callback) => {
  const events: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({ handleLLMNewToken: (token): void => { if (token) events.push(token); }, handleLLMEnd: (output): void => { events.push(JSON.stringify(output)); } });
  observer.awaitHandlers = true;
  const blockingWork = (): void => { const end = performance.now() + 35; while (performance.now() < end) { /* bounded synchronous fixture */ } };
  let inspected = false;
  const protection = policy({ timeoutMs: 10, classify: () => { if (callback === 'classify') blockingWork(); return 'prose'; }, inspect: () => { inspected = true; if (callback === 'inspect') blockingWork(); return approved('Late control'); } });
  await expect(invoke(new Transport([new AIMessageChunk(fixtures.canaries[0])]), protection, [observer])).rejects.toMatchObject({ code: 'timeout' });
  if (callback === 'classify') expect(inspected).toBe(false);
  expect(events).toEqual([]);
});

it.each(['orphan', 'wrong-index', 'wrong-args', 'alias'])('rejects unbound Anthropic input fragments (%s)', async (kind) => {
  const start = new AIMessageChunk({ content: [{ type: 'tool_use', index: 1, id: 'call-control', name: 'lookup', input: '' }], tool_call_chunks: [{ index: 1, id: 'call-control', name: 'lookup', args: '' }] });
  const delta = { type: 'tool_use', index: kind === 'wrong-index' ? 2 : 1, input: '{"count":42}', ...(kind === 'alias' ? { raw: fixtures.canaries[0] } : {}) };
  Reflect.deleteProperty(delta, 'type');
  const chunk = new AIMessageChunk({ content: [delta], tool_call_chunks: [{ index: delta.index, args: kind === 'wrong-args' ? '{"count":43}' : delta.input }] });
  await expect(invoke(new Transport(kind === 'orphan' ? [chunk] : [start, chunk]), policy())).rejects.toMatchObject({ code: 'unsupported' });
});

it('bounds fragments within a coalesced text event', async () => {
  const content = Array.from({ length: 100 }, () => ({ type: 'text', index: 0, text: 'x' }));
  await expect(invoke(new Transport([new AIMessageChunk({ content })]), policy({ maxAttemptBytes: 4096 }))).rejects.toMatchObject({ code: 'overflow' });
});

it('bounds admitted tool-index bookkeeping within a coalesced control event', async () => {
  const content = Array.from({ length: 100 }, (_value, index) => ({ type: 'tool_use', index, id: `call-${index}`, name: 'lookup', input: '' }));
  const tool_call_chunks = content.map((block) => ({ index: block.index, id: block.id, name: block.name, args: '' }));
  await expect(invoke(new Transport([new AIMessageChunk({ content, tool_call_chunks })]), policy({ maxAttemptBytes: 4096 }))).rejects.toMatchObject({ code: 'overflow' });
});

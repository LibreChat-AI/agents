import { AsyncLocalStorage } from 'node:async_hooks';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type {
  BindToolsInput,
  BaseChatModelCallOptions,
} from '@langchain/core/language_models/chat_models';
import type { ChatResult, LLMResult } from '@langchain/core/outputs';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { HostModelDescriptor, HostModelRequest } from './model';
import { createHostModelService, HostModelAdmission } from './modelService';
import { FakeChatModel } from '@/llm/fake';
import { Run, Providers } from '../index';
import { HostChatModel } from './model';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
  tenantId: 'tenant',
  conversationId: 'thread',
};
const tool = {
  type: 'function' as const,
  function: {
    name: 'echo',
    parameters: { type: 'object', properties: { text: { type: 'string' } } },
  },
};
const descriptor: HostModelDescriptor = {
  version: 1,
  reference: 'model',
  provider: 'openAI',
  model: 'approved',
  streaming: true,
  tools: [tool],
};
const request = (): HostModelRequest => ({
  version: 1,
  identity,
  reference: 'model',
  messages: [new HumanMessage('hello')],
  toolNames: [],
});
const limits = {
  timeoutMs: 5000,
  maxRequestBytes: 65536,
  maxResponseBytes: 65536,
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
const flush = (): Promise<void> =>
  new Promise((resolve) => setImmediate(resolve));

class Model extends FakeChatModel {
  readonly entered = deferred<void>();
  blocked?: Promise<void>;
  inputs: BaseMessage[][] = [];
  selections: {
    tools: BindToolsInput[];
    options?: Partial<BaseChatModelCallOptions>;
  }[] = [];
  signal?: AbortSignal;
  output = 'host result';
  cleanup = deferred<void>();
  cleanupGate?: Promise<void>;
  chunks = ['one', 'two'];
  streamEntered = deferred<void>();
  constructor() {
    super({ responses: ['unused'] });
  }
  override bindTools(
    tools: BindToolsInput[],
    options?: Partial<BaseChatModelCallOptions>
  ): ReturnType<Model['withConfig']> {
    this.selections.push({ tools, options });
    return this.withConfig(options ?? {});
  }
  override async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): Promise<ChatResult> {
    this.inputs.push(messages);
    this.signal = options.signal;
    this.entered.resolve();
    await this.blocked;
    return {
      generations: [{ text: this.output, message: new AIMessage(this.output) }],
    };
  }
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): AsyncGenerator<ChatGenerationChunk> {
    this.inputs.push(messages);
    this.signal = options.signal;
    this.streamEntered.resolve();
    try {
      for (const text of this.chunks) {
        await this.blocked;
        yield new ChatGenerationChunk({
          text,
          message: new AIMessageChunk(text),
        });
      }
    } finally {
      await this.cleanupGate;
      this.cleanup.resolve();
    }
  }
}
function fixture(
  options: {
    model?: Model;
    admission?: HostModelAdmission;
    timeoutMs?: number;
    maxRequestBytes?: number;
    maxResponseBytes?: number;
    hostSignal?: AbortSignal;
    callbacks?: RunnableConfig['callbacks'];
  } = {}
) {
  const model = options.model ?? new Model();
  const admission =
    options.admission ??
    new HostModelAdmission({ maxConcurrentCalls: 2, maxRequestBytes: 131072 });
  const bound = createHostModelService({
    identity,
    descriptor,
    model,
    admission,
    limits: {
      ...limits,
      ...(options.timeoutMs !== undefined
        ? { timeoutMs: options.timeoutMs }
        : {}),
      ...(options.maxRequestBytes !== undefined
        ? { maxRequestBytes: options.maxRequestBytes }
        : {}),
      ...(options.maxResponseBytes !== undefined
        ? { maxResponseBytes: options.maxResponseBytes }
        : {}),
    },
    config: {
      signal: options.hostSignal,
      metadata: { host: true },
      callbacks: options.callbacks,
    },
  });
  return { ...bound, model, admission };
}

it('runs a native model with host configuration through the consumer facade', async () => {
  const { model, service, admission } = fixture();
  const hostStart = jest.fn();
  model.callbacks = [{ handleLLMStart: hostStart }];
  const worker = new HostChatModel({ identity, descriptor, service });
  const answer = await worker
    .bindTools([tool], { tool_choice: 'echo' })
    .invoke('hi', { configurable: { model: 'foreign', apiKey: 'PRIVATE' } });
  expect(answer.text).toBe('host result');
  expect(model.selections).toEqual([
    {
      tools: [tool],
      options: {
        tool_choice: { type: 'function', function: { name: 'echo' } },
      },
    },
  ]);
  expect(hostStart).toHaveBeenCalledTimes(1);
  expect(admission.snapshot()).toEqual({
    calls: 0,
    requestBytes: 0,
    retainedCalls: 0,
    retainedRequestBytes: 0,
  });
});

it.each([
  'runtimeId',
  'agentId',
  'runId',
  'userId',
  'tenantId',
  'conversationId',
] as const)('rejects a foreign %s before calling the model', async (field) => {
  const { service, model, admission } = fixture();
  await expect(
    service.invoke({
      ...request(),
      identity: { ...identity, [field]: 'foreign' },
    })
  ).rejects.toMatchObject({ code: 'forbidden' });
  expect(model.inputs).toHaveLength(0);
  expect(admission.snapshot().calls).toBe(0);
});

it.each([
  { reference: 'foreign' },
  { toolNames: ['foreign'] },
  { toolNames: ['echo', 'echo'] },
  { toolNames: [], toolChoice: { mode: 'required' as const } },
  {
    toolNames: ['echo'],
    toolChoice: { mode: 'tool' as const, name: 'foreign' },
  },
])('rejects unauthorized reference or tool selection %j', async (overrides) => {
  const { service, model } = fixture();
  await expect(
    service.invoke({ ...request(), ...overrides })
  ).rejects.toMatchObject({ code: 'forbidden' });
  expect(model.inputs).toHaveLength(0);
  expect(model.selections).toHaveLength(0);
});

it('snapshots messages before asynchronous dispatch and keeps descriptor changes private', async () => {
  const { service, model, descriptor: returned } = fixture();
  const message = new HumanMessage({
    content: [{ type: 'text', text: 'original' }],
    additional_kwargs: { marker: 'original' },
  });
  const result = service.invoke({
    ...request(),
    messages: [message],
    toolNames: ['echo'],
  });
  message.content = 'changed';
  message.additional_kwargs.marker = 'changed';
  returned.tools[0].function.name = 'changed';
  await result;
  expect(model.inputs[0][0].content).toEqual([
    { type: 'text', text: 'original' },
  ]);
  expect(model.inputs[0][0].additional_kwargs.marker).toBe('original');
  expect(model.selections[0].tools).toEqual([tool]);
});

it('bounds requests before model execution and enforces shared admission across bindings', async () => {
  const admission = new HostModelAdmission({
    maxConcurrentCalls: 1,
    maxRequestBytes: 1024,
  });
  const gate = deferred<void>();
  const first = fixture({ admission });
  first.model.blocked = gate.promise;
  const second = fixture({ admission });
  const pending = first.service.invoke(request());
  await first.model.entered.promise;
  await expect(second.service.invoke(request())).rejects.toMatchObject({
    code: 'overflow',
  });
  expect(second.model.inputs).toHaveLength(0);
  gate.resolve();
  await pending;
  await expect(
    second.service.invoke({
      ...request(),
      messages: [new HumanMessage('x'.repeat(2048))],
    })
  ).rejects.toMatchObject({ code: 'overflow' });
  expect(admission.snapshot().calls).toBe(0);
});

it.each(['cancel', 'timeout'] as const)(
  'retains an invocation after %s until an ignoring provider settles',
  async (mode) => {
    const controller = new AbortController();
    const gate = deferred<void>();
    const { service, model, admission } = fixture({
      timeoutMs: mode === 'timeout' ? 20 : 5000,
    });
    model.blocked = gate.promise;
    const result = service.invoke(request(), controller.signal);
    const assertion = expect(result).rejects.toMatchObject({
      code: mode === 'timeout' ? 'timeout' : 'cancelled',
    });
    await model.entered.promise;
    if (mode === 'cancel') controller.abort();
    await assertion;
    expect(model.signal?.aborted).toBe(true);
    expect(admission.snapshot()).toMatchObject({ calls: 1, retainedCalls: 1 });
    gate.resolve();
    await flush();
    expect(admission.snapshot()).toEqual({
      calls: 0,
      requestBytes: 0,
      retainedCalls: 0,
      retainedRequestBytes: 0,
    });
  }
);

it('never lets a worker signal override host execution cancellation', async () => {
  const host = new AbortController();
  const worker = new AbortController();
  const gate = deferred<void>();
  const { service, model, admission } = fixture({ hostSignal: host.signal });
  model.blocked = gate.promise;
  const result = service.invoke(request(), worker.signal);
  const assertion = expect(result).rejects.toMatchObject({ code: 'cancelled' });
  await model.entered.promise;
  host.abort();
  await assertion;
  expect(worker.signal.aborted).toBe(false);
  expect(model.signal?.aborted).toBe(true);
  gate.resolve();
  await flush();
  expect(admission.snapshot().calls).toBe(0);
});

it('retains a cancelled stream until its outstanding provider read and cleanup drain', async () => {
  const controller = new AbortController();
  const gate = deferred<void>();
  const cleanup = deferred<void>();
  const { service, model, admission } = fixture();
  model.blocked = gate.promise;
  model.cleanupGate = cleanup.promise;
  const iterable = service.stream(request(), controller.signal);
  const stream = iterable[Symbol.asyncIterator]();
  const pending = stream.next();
  const assertion = expect(pending).rejects.toMatchObject({
    code: 'cancelled',
  });
  await model.streamEntered.promise;
  controller.abort();
  await assertion;
  expect(admission.snapshot()).toMatchObject({ calls: 1, retainedCalls: 1 });
  gate.resolve();
  await flush();
  expect(admission.snapshot().calls).toBe(1);
  cleanup.resolve();
  await model.cleanup.promise;
  await flush();
  expect(admission.snapshot().calls).toBe(0);
});

it('times out and closes a stream even while its consumer is paused after a chunk', async () => {
  const { service, model, admission } = fixture({ timeoutMs: 20 });
  const stream = service.stream(request())[Symbol.asyncIterator]();
  expect((await stream.next()).value?.text).toBe('one');
  await new Promise((resolve) => setTimeout(resolve, 40));
  await model.cleanup.promise;
  await flush();
  expect(model.signal?.aborted).toBe(true);
  expect(admission.snapshot().calls).toBe(0);
  await expect(stream.next()).rejects.toMatchObject({ code: 'timeout' });
});

it('cleans up early consumer return without cancelling other bindings', async () => {
  const first = fixture();
  const second = fixture();
  const stream = first.service.stream(request())[Symbol.asyncIterator]();
  await stream.next();
  await stream.return?.();
  await first.model.cleanup.promise;
  await flush();
  expect(first.admission.snapshot().calls).toBe(0);
  await expect(second.service.invoke(request())).resolves.toMatchObject({
    content: 'host result',
  });
});

it.each(['invoke', 'stream'] as const)(
  'bounds %s responses including serialized metadata',
  async (mode) => {
    const { service, model, admission } = fixture({ maxResponseBytes: 200 });
    model.output = 'x'.repeat(500);
    model.chunks = ['x'.repeat(500)];
    const result =
      mode === 'invoke'
        ? service.invoke(request())
        : service.stream(request())[Symbol.asyncIterator]().next();
    await expect(result).rejects.toMatchObject({ code: 'overflow' });
    await flush();
    expect(admission.snapshot().calls).toBe(0);
  }
);

it('caps cumulative stream size across otherwise valid chunks', async () => {
  const { service, model, admission } = fixture({ maxResponseBytes: 600 });
  model.chunks = Array.from({ length: 20 }, () => 'small');
  const result = (async () => {
    for await (const chunk of service.stream(request())) void chunk;
  })();
  await expect(result).rejects.toMatchObject({ code: 'overflow' });
  await flush();
  expect(admission.snapshot().calls).toBe(0);
});

it('sanitizes failures, observes late rejection, and does not retry', async () => {
  const { service, model, admission } = fixture();
  const gate = deferred<void>();
  model.blocked = gate.promise;
  const pending = service.invoke(request());
  const assertion = expect(pending).rejects.toMatchObject({
    message: 'Host model service: unavailable',
    code: 'unavailable',
  });
  await model.entered.promise;
  gate.reject(new Error('PRIVATE credential'));
  await assertion;
  expect(model.inputs).toHaveLength(1);
  expect(admission.snapshot().calls).toBe(0);
});

it.each([0, -1, Infinity, 1.5])('rejects invalid host limits %s', (value) => {
  expect(
    () =>
      new HostModelAdmission({
        maxConcurrentCalls: value,
        maxRequestBytes: 1000,
      })
  ).toThrow();
  expect(() =>
    createHostModelService({
      identity,
      descriptor,
      model: new Model(),
      admission: new HostModelAdmission({
        maxConcurrentCalls: 1,
        maxRequestBytes: 1000,
      }),
      limits: { ...limits, timeoutMs: value },
    })
  ).toThrow();
});

it('explicitly binds an empty selected tool set rather than inheriting the model defaults', async () => {
  const { model, service } = fixture();
  await service.invoke({ ...request(), toolChoice: { mode: 'none' } });
  expect(model.selections).toEqual([
    { tools: [], options: { tool_choice: 'none' } },
  ]);
});

it('preserves native rich content, fragments, usage and host streaming callbacks', async () => {
  class RichModel extends Model {
    override async *_streamResponseChunks(): AsyncGenerator<ChatGenerationChunk> {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: [{ type: 'reasoning', reasoning: 'thought' }],
          tool_call_chunks: [
            {
              id: 'call',
              name: 'echo',
              args: '{}',
              index: 0,
              type: 'tool_call_chunk',
            },
          ],
          usage_metadata: {
            input_tokens: 3,
            output_tokens: 2,
            total_tokens: 5,
          },
          response_metadata: { finish_reason: 'tool_calls' },
        }),
      });
    }
  }
  const model = new RichModel();
  const end = jest.fn();
  model.callbacks = [{ handleLLMEnd: end }];
  const { service, admission } = fixture({ model });
  const chunks = [];
  for await (const chunk of service.stream({
    ...request(),
    toolNames: ['echo'],
  }))
    chunks.push(chunk);
  expect(chunks[0].content).toEqual([
    { type: 'reasoning', reasoning: 'thought' },
  ]);
  expect(chunks[0].tool_call_chunks?.[0]).toMatchObject({
    id: 'call',
    name: 'echo',
    args: '{}',
  });
  expect(chunks[0].usage_metadata?.total_tokens).toBe(5);
  expect(chunks[0].response_metadata.finish_reason).toBe('tool_calls');
  expect(end).toHaveBeenCalledTimes(1);
  await flush();
  expect(admission.snapshot().calls).toBe(0);
});

it('does not prefetch the provider stream', async () => {
  let produced = 0;
  class PullModel extends Model {
    override async *_streamResponseChunks(): AsyncGenerator<ChatGenerationChunk> {
      for (let index = 0; index < 4; index++) {
        produced++;
        yield new ChatGenerationChunk({
          text: 'chunk',
          message: new AIMessageChunk('chunk'),
        });
      }
    }
  }
  const { service } = fixture({ model: new PullModel() });
  const stream = service.stream(request())[Symbol.asyncIterator]();
  await stream.next();
  await flush();
  expect(produced).toBe(1);
  await stream.return?.();
});

it('honors the byte quota independently of the per-call limit', async () => {
  const admission = new HostModelAdmission({
    maxConcurrentCalls: 5,
    maxRequestBytes: 300,
  });
  const { service, model } = fixture({ admission });
  const gate = deferred<void>();
  model.blocked = gate.promise;
  const first = service.invoke({
    ...request(),
    messages: [new HumanMessage('a'.repeat(80))],
  });
  await model.entered.promise;
  await expect(
    service.invoke({
      ...request(),
      messages: [new HumanMessage('b'.repeat(80))],
    })
  ).rejects.toMatchObject({ code: 'overflow' });
  expect(admission.snapshot().calls).toBe(1);
  gate.resolve();
  await first;
  expect(admission.snapshot().calls).toBe(0);
});

it('rejects full capacity before serializing another request', async () => {
  const admission = new HostModelAdmission({
    maxConcurrentCalls: 1,
    maxRequestBytes: 65536,
  });
  const { model, service } = fixture({ admission });
  const gate = deferred<void>();
  model.blocked = gate.promise;
  const pending = service.invoke(request());
  await model.entered.promise;
  const message = new HumanMessage('second');
  const serialize = jest.spyOn(message, 'getType');
  await expect(
    service.invoke({ ...request(), messages: [message] })
  ).rejects.toMatchObject({ code: 'overflow' });
  expect(serialize).not.toHaveBeenCalled();
  gate.resolve();
  await pending;
});

it('rolls back capacity after malformed message serialization', async () => {
  const { service, admission } = fixture();
  const message = new HumanMessage('hello');
  message.additional_kwargs.invalid = () => 'not data';
  await expect(
    service.invoke({ ...request(), messages: [message] })
  ).rejects.toMatchObject({ code: 'incompatible' });
  expect(admission.snapshot().calls).toBe(0);
  await expect(service.invoke(request())).resolves.toMatchObject({
    content: 'host result',
  });
});

it('enforces the minimum message framing bytes for an empty request', async () => {
  const { service, admission } = fixture({ maxRequestBytes: 1 });
  await expect(
    service.invoke({ ...request(), messages: [] })
  ).rejects.toMatchObject({ code: 'overflow' });
  expect(admission.snapshot().calls).toBe(0);
});

it.each([
  ['openAI', { type: 'function', function: { name: 'echo' } }],
  ['anthropic', { type: 'tool', name: 'echo' }],
  ['bedrock', { tool: { name: 'echo' } }],
  ['google', 'any'],
])('uses native forced-tool choice for %s', async (provider, choice) => {
  const model = new Model();
  const { service } = createHostModelService({
    identity,
    descriptor: { ...descriptor, provider: String(provider) },
    model,
    admission: new HostModelAdmission({
      maxConcurrentCalls: 1,
      maxRequestBytes: 65536,
    }),
    limits,
  });
  await service.invoke({
    ...request(),
    toolNames: ['echo'],
    toolChoice: { mode: 'tool', name: 'echo' },
  });
  expect(model.selections[0].options).toMatchObject({ tool_choice: choice });
  if (provider === 'google')
    expect(model.selections[0].options).toMatchObject({
      allowedFunctionNames: ['echo'],
      allowed_function_names: ['echo'],
    });
});

it('uses the portable required-choice alias and removes tools for Bedrock none', async () => {
  const model = new Model();
  const { service } = createHostModelService({
    identity,
    descriptor: { ...descriptor, provider: 'bedrock' },
    model,
    admission: new HostModelAdmission({
      maxConcurrentCalls: 1,
      maxRequestBytes: 65536,
    }),
    limits,
  });
  await service.invoke({
    ...request(),
    toolNames: ['echo'],
    toolChoice: { mode: 'required' },
  });
  await service.invoke({
    ...request(),
    toolNames: ['echo'],
    toolChoice: { mode: 'none' },
  });
  expect(model.selections[0].options).toEqual({ tool_choice: 'any' });
  expect(model.selections[1]).toEqual({ tools: [], options: undefined });
});

it('immediately reuses capacity after a fully consumed stream', async () => {
  const admission = new HostModelAdmission({
    maxConcurrentCalls: 1,
    maxRequestBytes: 65536,
  });
  const { service } = fixture({ admission });
  for await (const chunk of service.stream(request())) void chunk;
  expect(admission.snapshot().calls).toBe(0);
  await expect(service.invoke(request())).resolves.toMatchObject({
    content: 'host result',
  });
});

it('runs the authenticated native binding through a real SDK graph', async () => {
  const { service, model, admission } = fixture();
  const run = await Run.create({
    runId: identity.runId,
    returnContent: true,
    tokenCounter: () => 1,
    graphConfig: {
      type: 'standard',
      llmConfig: { provider: Providers.OPENAI },
      agents: [
        { agentId: identity.agentId, provider: Providers.OPENAI, tools: [] },
      ],
    },
  });
  run.Graph!.overrideModel = new HostChatModel({
    identity,
    descriptor,
    service,
  });
  const result = await run.processStream(
    { messages: [new HumanMessage('hello')] },
    { version: 'v2', configurable: { thread_id: 'thread' } }
  );
  expect(JSON.stringify(result)).toContain('onetwo');
  expect(model.inputs).toHaveLength(1);
  expect(admission.snapshot().calls).toBe(0);
});

it('preserves binary and structured message metadata without JSON coercion', async () => {
  const { service, model } = fixture();
  const bytes = Uint8Array.from([1, 2, 3]);
  const date = new Date('2026-01-01T00:00:00.000Z');
  const message = new HumanMessage({
    content: 'hello',
    additional_kwargs: {
      bytes,
      date,
      large: 123n,
      values: new Map([['a', 1]]),
      flags: new Set(['a']),
    },
  });
  const result = service.invoke({ ...request(), messages: [message] });
  bytes[0] = 9;
  await result;
  expect(model.inputs[0][0].additional_kwargs).toEqual({
    bytes: Uint8Array.from([1, 2, 3]),
    date,
    large: 123n,
    values: new Map([['a', 1]]),
    flags: new Set(['a']),
  });
});

it('preserves tool-result artifacts and status in the native model input', async () => {
  const { service, model } = fixture();
  const message = new ToolMessage({
    content: 'tool output',
    tool_call_id: 'call',
    status: 'error',
    artifact: { data: Uint8Array.from([4, 5]) },
  });
  await service.invoke({ ...request(), messages: [message] });
  expect(model.inputs[0][0]).toMatchObject({
    tool_call_id: 'call',
    status: 'error',
    artifact: { data: Uint8Array.from([4, 5]) },
  });
});

function callbackLog(waitForEnd?: Promise<void>) {
  const events: {
    type: 'start' | 'end' | 'error';
    runId: string;
    output?: LLMResult;
    parentRunId?: string;
    tags?: string[];
  }[] = [];
  const handler = BaseCallbackHandler.fromMethods({
    handleChatModelStart: (
      _model,
      _messages,
      runId,
      parentRunId,
      _extra,
      tags
    ): void => {
      events.push({ type: 'start', runId, parentRunId, tags });
    },
    handleLLMEnd: async (output, runId, parentRunId, tags): Promise<void> => {
      events.push({ type: 'end', runId, output, parentRunId, tags });
      await waitForEnd;
    },
    handleLLMError: (_error, runId): void => {
      events.push({ type: 'error', runId });
    },
  });
  handler.awaitHandlers = true;
  return { handler, events };
}

it.each(['return', 'timeout', 'overflow'] as const)(
  'closes model and host-config callbacks exactly once after early %s',
  async (mode) => {
    const modelLog = callbackLog();
    const configLog = callbackLog();
    const { service, model, admission } = fixture({
      timeoutMs: mode === 'timeout' ? 20 : 5000,
      maxResponseBytes: 1024,
      callbacks: [configLog.handler],
    });
    model.callbacks = [modelLog.handler];
    model.tags = ['host-model-tag'];
    if (mode === 'overflow') model.chunks = ['one', 'x'.repeat(2000)];
    const stream = service.stream(request())[Symbol.asyncIterator]();
    expect((await stream.next()).value?.text).toBe('one');
    if (mode === 'return') await stream.return?.();
    if (mode === 'timeout')
      await new Promise((resolve) => setTimeout(resolve, 40));
    if (mode === 'overflow')
      await expect(stream.next()).rejects.toMatchObject({ code: 'overflow' });
    await model.cleanup.promise;
    await flush();
    for (const log of [modelLog, configLog]) {
      expect(log.events.map((event) => event.type)).toEqual(['start', 'end']);
      expect(log.events[1].runId).toBe(log.events[0].runId);
      expect(log.events[0].tags).toContain('host-model-tag');
      expect(log.events[1].tags).toEqual(log.events[0].tags);
      expect(log.events[1].parentRunId).toBe(log.events[0].parentRunId);
      expect(log.events[1].output?.generations[0][0].text).toBe('one');
      expect(log.events[1].output?.llmOutput).toMatchObject({
        controlFlow: 'HostModelTerminated',
        reason: mode === 'return' ? 'cancelled' : mode,
        usage_incomplete: true,
      });
    }
    expect(modelLog.events[0].runId).toBe(configLog.events[0].runId);
    expect(admission.snapshot().calls).toBe(0);
    if (mode === 'timeout')
      await expect(stream.next()).rejects.toMatchObject({ code: 'timeout' });
  }
);

it('does not duplicate the native terminal callback on success or provider failure', async () => {
  for (const fail of [false, true]) {
    const log = callbackLog();
    const { service, model, admission } = fixture();
    model.callbacks = [log.handler];
    const gate = deferred<void>();
    model.blocked = gate.promise;
    const result = (async () => {
      for await (const chunk of service.stream(request())) void chunk;
    })();
    const assertion = fail
      ? expect(result).rejects.toMatchObject({ code: 'unavailable' })
      : expect(result).resolves.toBeUndefined();
    await model.streamEntered.promise;
    if (fail) gate.reject(new Error('provider failed'));
    else gate.resolve();
    await assertion;
    await flush();
    expect(log.events.map((event) => event.type)).toEqual([
      'start',
      fail ? 'error' : 'end',
    ]);
    expect(log.events[1].runId).toBe(log.events[0].runId);
    expect(admission.snapshot().calls).toBe(0);
  }
});

it('does not synthesize another end after native cancellation closes the callback run', async () => {
  const log = callbackLog();
  const { service, model, admission } = fixture();
  model.callbacks = [log.handler];
  const controller = new AbortController();
  const gate = deferred<void>();
  model.blocked = gate.promise;
  const iterable = service.stream(request(), controller.signal);
  const pending = iterable[Symbol.asyncIterator]().next();
  const assertion = expect(pending).rejects.toMatchObject({
    code: 'cancelled',
  });
  await model.streamEntered.promise;
  controller.abort();
  await assertion;
  gate.resolve();
  await model.cleanup.promise;
  await flush();
  expect(log.events.map((event) => event.type)).toEqual(['start', 'error']);
  expect(admission.snapshot().calls).toBe(0);
});

it('retains admission while an early-close host callback is still running', async () => {
  const gate = deferred<void>();
  const log = callbackLog(gate.promise);
  const { service, model, admission } = fixture();
  model.callbacks = [log.handler];
  const stream = service.stream(request())[Symbol.asyncIterator]();
  await stream.next();
  await stream.return?.();
  await model.cleanup.promise;
  await flush();
  expect(log.events.map((event) => event.type)).toEqual(['start', 'end']);
  expect(admission.snapshot()).toMatchObject({ calls: 1, retainedCalls: 1 });
  gate.resolve();
  await flush();
  expect(admission.snapshot().calls).toBe(0);
});

it.each(['invoke', 'stream'] as const)(
  'isolates %s from the caller ambient LangChain configuration',
  async (mode) => {
    AsyncLocalStorageProviderSingleton.initializeGlobalInstance(
      new AsyncLocalStorage()
    );
    const foreignStart = jest.fn();
    const hostStart = jest.fn();
    const { service, model } = fixture();
    model.callbacks = [{ handleChatModelStart: hostStart }];
    await AsyncLocalStorageProviderSingleton.runWithConfig(
      {
        metadata: { foreign: 'PRIVATE' },
        configurable: { model: 'foreign-model' },
        callbacks: [{ handleChatModelStart: foreignStart }],
      },
      async () => {
        expect(
          AsyncLocalStorageProviderSingleton.getRunnableConfig().metadata
            .foreign
        ).toBe('PRIVATE');
        if (mode === 'invoke') await service.invoke(request());
        else {
          for await (const chunk of service.stream(request())) void chunk;
        }
      }
    );
    expect(foreignStart).not.toHaveBeenCalled();
    expect(hostStart).toHaveBeenCalledTimes(1);
    expect(hostStart.mock.calls[0][6]).toMatchObject({ host: true });
    expect(JSON.stringify(hostStart.mock.calls[0][6])).not.toContain('PRIVATE');
    expect(JSON.stringify(hostStart.mock.calls[0][6])).not.toContain(
      'foreign-model'
    );
  }
);

it('preserves measured provider usage when synthesizing an early callback end', async () => {
  class UsageModel extends Model {
    override async *_streamResponseChunks(): AsyncGenerator<ChatGenerationChunk> {
      yield new ChatGenerationChunk({
        text: 'partial',
        message: new AIMessageChunk({
          content: 'partial',
          usage_metadata: {
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
          },
        }),
      });
      yield new ChatGenerationChunk({
        text: 'more',
        message: new AIMessageChunk('more'),
      });
    }
  }
  const log = callbackLog();
  const model = new UsageModel();
  model.callbacks = [log.handler];
  const { service, admission } = fixture({ model });
  const stream = service.stream(request())[Symbol.asyncIterator]();
  await stream.next();
  await stream.return?.();
  await flush();
  expect(log.events.map((event) => event.type)).toEqual(['start', 'end']);
  expect(log.events[1].output?.generations[0][0]).toMatchObject({
    message: {
      usage_metadata: { input_tokens: 7, output_tokens: 3, total_tokens: 10 },
    },
  });
  expect(log.events[1].output?.llmOutput).toMatchObject({
    usage_incomplete: false,
  });
  expect(admission.snapshot().calls).toBe(0);
});

it('closes the native run even when partial chunks cannot be concatenated', async () => {
  class InvalidChunksModel extends Model {
    override async *_streamResponseChunks(): AsyncGenerator<ChatGenerationChunk> {
      for (const value of ['PRIVATE', 42]) {
        yield new ChatGenerationChunk({
          text: 'part',
          message: new AIMessageChunk({
            content: 'part',
            response_metadata: { conflicting: value },
          }),
        });
      }
    }
  }
  const model = new InvalidChunksModel();
  const log = callbackLog();
  model.callbacks = [log.handler];
  const { service, admission } = fixture({ model });
  const stream = service.stream(request())[Symbol.asyncIterator]();
  await stream.next();
  await stream.next();
  await stream.return?.();
  await flush();
  expect(log.events.map((event) => event.type)).toEqual(['start', 'error']);
  expect(log.events[1].runId).toBe(log.events[0].runId);
  expect(admission.snapshot().calls).toBe(0);
});

it.each(['cumulative', 'single', 'split-usage'] as const)(
  'preserves a usage-only tail rejected by the %s response limit',
  async (mode) => {
    class TailUsageModel extends Model {
      override async *_streamResponseChunks(): AsyncGenerator<ChatGenerationChunk> {
        yield new ChatGenerationChunk({
          text: 'a'.repeat(200),
          message: new AIMessageChunk({
            content: 'a'.repeat(200),
            ...(mode === 'split-usage'
              ? {
                usage_metadata: {
                  input_tokens: 7,
                  output_tokens: 0,
                  total_tokens: 7,
                },
              }
              : {}),
          }),
        });
        yield new ChatGenerationChunk({
          text: '',
          message: new AIMessageChunk({
            content: '',
            usage_metadata: {
              input_tokens: mode === 'split-usage' ? 0 : 7,
              output_tokens: 3,
              total_tokens: mode === 'split-usage' ? 3 : 10,
              input_token_details: { cache_read: 2 },
              output_token_details: { reasoning: 1 },
            },
            response_metadata: {
              padding: mode === 'single' ? 'x'.repeat(2000) : '',
            },
          }),
        });
      }
    }
    const log = callbackLog();
    const model = new TailUsageModel();
    model.callbacks = [log.handler];
    const { service, admission } = fixture({ model, maxResponseBytes: 600 });
    const stream = service.stream(request())[Symbol.asyncIterator]();
    expect((await stream.next()).value?.text).toBe('a'.repeat(200));
    await expect(stream.next()).rejects.toMatchObject({ code: 'overflow' });
    await flush();
    expect(log.events.map((event) => event.type)).toEqual(['start', 'end']);
    expect(log.events[1].output?.generations[0][0]).toMatchObject({
      text: 'a'.repeat(200),
      message: {
        usage_metadata: {
          input_tokens: 7,
          output_tokens: 3,
          total_tokens: 10,
          input_token_details: { cache_read: 2 },
          output_token_details: { reasoning: 1 },
        },
      },
    });
    expect(log.events[1].output?.llmOutput).toMatchObject({
      reason: 'overflow',
      usage_incomplete: false,
      tokenUsage: { promptTokens: 7, completionTokens: 3, totalTokens: 10 },
    });
    expect(admission.snapshot().calls).toBe(0);
  }
);

import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type {
  HostModelDescriptor,
  HostModelRequest,
  HostModelService,
} from './model';
import type { RuntimeProviderName } from '@/types';
import { Run, Providers, FakeChatModel, GraphEvents } from '../index';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { HostChatModel, HostModelServiceError } from './model';
import { registerProvider } from '../provider-registration';
import { createHostToolBindings } from './index';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
  tenantId: 'tenant',
};
const tool = {
  type: 'function' as const,
  function: {
    name: 'echo',
    description: 'Echo',
    parameters: {
      type: 'object' as const,
      properties: { text: { type: 'string' as const } },
      required: ['text'],
    },
  },
};
const descriptor: HostModelDescriptor = {
  version: 1,
  reference: 'host-model',
  provider: 'openAI',
  model: 'host-configured-model',
  streaming: true,
  tools: [tool],
};
const messages = [new HumanMessage('hello')];
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
function fixture(
  overrides: Partial<HostModelService> = {},
  config = descriptor
) {
  const requests: HostModelRequest[] = [];
  const native = new FakeChatModel({ responses: ['hello'], sleep: 0 });
  const service: HostModelService = {
    version: 1,
    invoke: async (request, signal) => {
      requests.push(request);
      return native.invoke([...request.messages], { signal });
    },
    stream: async function* (request, signal) {
      requests.push(request);
      yield* await native.stream([...request.messages], { signal });
    },
    ...overrides,
  };
  return {
    model: new HostChatModel({ identity, descriptor: config, service }),
    requests,
    service,
    native,
  };
}

it('invokes the actual host model without forwarding worker configuration or credentials', async () => {
  const { model, requests } = fixture();
  const answer = await model.invoke(messages, {
    configurable: { apiKey: 'PRIVATE', userId: 'other', temperature: 9 },
    metadata: { secret: 'PRIVATE' },
    stop: ['hello'],
  });
  expect(answer.text).toBe('hello');
  expect(requests).toEqual([
    {
      version: 1,
      identity,
      reference: descriptor.reference,
      messages,
      toolNames: [],
    },
  ]);
  expect(JSON.stringify(model.toJSON())).not.toContain('user');
  expect(model.getLsParams()).toEqual({
    ls_model_type: 'chat',
    ls_provider: 'openAI',
    ls_model_name: descriptor.model,
  });
});

it('snapshots identity and approved schemas when the model is bound', async () => {
  const mutableIdentity = { ...identity };
  const mutableDescriptor = structuredClone(descriptor);
  const { service, requests } = fixture();
  const model = new HostChatModel({
    identity: mutableIdentity,
    descriptor: mutableDescriptor,
    service,
  });
  mutableIdentity.userId = 'other';
  mutableDescriptor.tools[0].function.name = 'foreign';
  await model.bindTools([tool]).invoke(messages);
  expect(requests[0].identity.userId).toBe('user');
  expect(Object.isFrozen(requests[0].identity)).toBe(true);
  expect(requests[0].toolNames).toEqual(['echo']);
});

it.each([
  'auto',
  'none',
  'required',
  'any',
  'echo',
  { type: 'function' as const, function: { name: 'echo' } },
  { type: 'tool' as const, name: 'echo' },
])('normalizes host-approved tool choice %j', async (choice) => {
  const { model, requests } = fixture();
  await model.bindTools([tool], { tool_choice: choice }).invoke(messages);
  expect(requests[0].toolChoice).toEqual(
    typeof choice !== 'string' || choice === 'echo'
      ? { mode: 'tool', name: 'echo' }
      : { mode: choice === 'any' ? 'required' : choice }
  );
});

it('compares tool schemas structurally and refuses foreign or changed schemas', async () => {
  const { model, requests } = fixture();
  await model
    .bindTools([
      {
        function: {
          parameters: tool.function.parameters,
          description: 'Echo',
          name: 'echo',
        },
        type: 'function',
      },
    ])
    .invoke(messages);
  expect(() =>
    model.bindTools([
      { ...tool, function: { ...tool.function, description: 'changed' } },
    ])
  ).toThrow(HostModelServiceError);
  expect(() => model.bindTools([tool, tool])).toThrow(HostModelServiceError);
  expect(() => model.bindTools([tool], { tool_choice: 'foreign' })).toThrow(
    HostModelServiceError
  );
  expect(() => model.bindTools([], { tool_choice: 'required' })).toThrow(
    HostModelServiceError
  );
  expect(requests).toHaveLength(1);
});

it('checks runtime tool selection before contacting the service', async () => {
  const { model, requests } = fixture();
  await expect(
    model.invoke(messages, { hostToolNames: ['foreign'] })
  ).rejects.toMatchObject({ code: 'forbidden' });
  await expect(
    model.invoke(messages, {
      hostToolNames: ['echo'],
      tool_choice: 'other',
    })
  ).rejects.toMatchObject({ code: 'forbidden' });
  expect(requests).toHaveLength(0);
});

it('preserves rich content, tool fragments, usage and provider metadata in native streaming', async () => {
  const chunks = [
    new AIMessageChunk({
      content: [{ type: 'reasoning', reasoning: 'thinking' }],
      tool_call_chunks: [
        {
          id: 'call',
          name: 'echo',
          args: '{"text":',
          index: 0,
          type: 'tool_call_chunk',
        },
      ],
    }),
    new AIMessageChunk({
      content: 'done',
      tool_call_chunks: [{ args: '"ok"}', index: 0, type: 'tool_call_chunk' }],
      usage_metadata: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
      response_metadata: { model_name: 'actual', finish_reason: 'tool_calls' },
    }),
  ];
  const tokens: string[] = [];
  const { model } = fixture({
    stream: async function* () {
      yield* chunks;
    },
  });
  let result: AIMessageChunk | undefined;
  for await (const chunk of await model.bindTools([tool]).stream(messages, {
    callbacks: [
      {
        handleLLMNewToken: (token) => {
          tokens.push(token);
        },
      },
    ],
  }))
    result = result ? result.concat(chunk) : chunk;
  expect(result?.content).toEqual([
    { type: 'reasoning', reasoning: 'thinking' },
    { type: 'text', text: 'done' },
  ]);
  expect(result?.tool_calls).toEqual([
    { id: 'call', name: 'echo', args: { text: 'ok' }, type: 'tool_call' },
  ]);
  expect(result?.usage_metadata).toMatchObject(chunks[1].usage_metadata!);
  expect(result?.response_metadata).toEqual(chunks[1].response_metadata);
  expect(tokens).toEqual(['', 'done']);
});

it('uses invocation for a host model that does not stream', async () => {
  const stream = jest.fn<
    ReturnType<HostModelService['stream']>,
    Parameters<HostModelService['stream']>
  >();
  const { model, requests } = fixture(
    { stream },
    { ...descriptor, streaming: false }
  );
  const result = [];
  for await (const chunk of await model.stream(messages))
    result.push(chunk.text);
  expect(result).toEqual(['hello']);
  expect(requests).toHaveLength(1);
  expect(stream).not.toHaveBeenCalled();
});

it('does not prefetch and cancels host work when the consumer stops', async () => {
  let signal: AbortSignal | undefined;
  let pulls = 0;
  const close = jest.fn(async () => ({
    done: true as const,
    value: undefined,
  }));
  const { model } = fixture({
    stream: (_request, received) => {
      signal = received;
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => {
            pulls++;
            return { done: false, value: new AIMessageChunk('one') };
          },
          return: close,
        }),
      };
    },
  });
  const stream = model._streamResponseChunks(messages, {});
  await stream.next();
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(pulls).toBe(1);
  await stream.return(undefined);
  expect(signal?.aborted).toBe(true);
  expect(close).toHaveBeenCalledTimes(1);
});

it.each(['invoke', 'stream'] as const)(
  'cancels a blocked %s promptly and observes late rejection',
  async (mode) => {
    const controller = new AbortController();
    const entered = deferred<void>();
    const pending = deferred<AIMessage>();
    let hostSignal: AbortSignal | undefined;
    const { model } = fixture({
      invoke: async (_request, signal) => {
        hostSignal = signal;
        entered.resolve();
        return pending.promise;
      },
      stream: async function* (_request, signal) {
        hostSignal = signal;
        entered.resolve();
        await pending.promise;
        yield new AIMessageChunk('late');
      },
    });
    const result =
      mode === 'invoke'
        ? model.invoke(messages, { signal: controller.signal })
        : (async () => {
          for await (const chunk of await model.stream(messages, {
            signal: controller.signal,
          }))
            void chunk;
        })();
    const assertion = expect(result).rejects.toThrow();
    await entered.promise;
    controller.abort(new Error('PRIVATE cancellation'));
    await assertion;
    expect(hostSignal?.aborted).toBe(true);
    pending.reject(new Error('PRIVATE late failure'));
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
);

it('does not contact the host for an already cancelled call', async () => {
  const { model, requests } = fixture();
  await expect(
    model.invoke(messages, { signal: AbortSignal.abort() })
  ).rejects.toThrow();
  expect(requests).toHaveLength(0);
});

it.each(['blocked', 'timeout', 'overflow', 'forbidden'] as const)(
  'preserves safe %s codes across package copies without raw error data',
  async (code) => {
    const { model } = fixture({
      invoke: async () => {
        throw { name: 'HostModelServiceError', code, message: 'PRIVATE' };
      },
    });
    await expect(model.invoke(messages)).rejects.toMatchObject({
      code,
      message: `Host model service: ${code}`,
    });
  }
);

it('sanitizes unexpected host errors and does not retry an ambiguous invocation', async () => {
  const invoke = jest.fn(async () => {
    throw new Error('PRIVATE provider error');
  });
  const { model } = fixture({ invoke });
  await expect(model.invoke(messages)).rejects.toMatchObject({
    code: 'unavailable',
    message: 'Host model service: unavailable',
  });
  expect(invoke).toHaveBeenCalledTimes(1);
});

it('rejects a non-AI host response', async () => {
  const { model } = fixture({
    // @ts-expect-error Deliberately violate the service result contract.
    invoke: async () => new HumanMessage('invalid'),
  });
  await expect(model.invoke(messages)).rejects.toMatchObject({
    code: 'incompatible',
  });
});

it('runs the model facade through the real SDK graph and event handlers', async () => {
  const visible: string[] = [];
  const { model, requests } = fixture();
  const run = await Run.create({
    runId: identity.runId,
    returnContent: true,
    tokenCounter: () => 1,
    customHandlers: {
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: (_event, data): void => {
          visible.push(JSON.stringify(data));
        },
      },
    },
    graphConfig: {
      type: 'standard',
      llmConfig: { provider: Providers.OPENAI },
      agents: [
        { agentId: identity.agentId, provider: Providers.OPENAI, tools: [] },
      ],
    },
  });
  run.Graph!.overrideModel = model;
  const result = await run.processStream(
    { messages },
    { version: 'v2', configurable: { thread_id: 'thread' } }
  );
  expect(JSON.stringify(result)).toContain('hello');
  expect(visible.length).toBeGreaterThan(0);
  expect(requests).toHaveLength(1);
  expect(requests[0].identity).toEqual(identity);
});

it('retains host callbacks and emits worker callbacks once without forwarding callback objects', async () => {
  const { model, native, requests } = fixture();
  const hostStart = jest.fn();
  const workerStart = jest.fn();
  native.callbacks = [{ handleLLMStart: hostStart }];
  await model.invoke(messages, {
    callbacks: [{ handleLLMStart: workerStart }],
  });
  expect(hostStart).toHaveBeenCalledTimes(1);
  expect(workerStart).toHaveBeenCalledTimes(1);
  expect(requests[0]).not.toHaveProperty('callbacks');
});

it.each(['invoke', 'stream'] as const)(
  'preserves local SDK control-flow aborts for %s',
  async (mode) => {
    const entered = deferred<void>();
    const pending = deferred<AIMessage>();
    const controller = new AbortController();
    const reason = new PreparedSubagentError('Local delegated work');
    const { model } = fixture({
      invoke: async () => {
        entered.resolve();
        return pending.promise;
      },
      stream: async function* () {
        entered.resolve();
        await pending.promise;
        yield new AIMessageChunk('late');
      },
    });
    const result =
      mode === 'invoke'
        ? model._generate(messages, { signal: controller.signal })
        : model
          ._streamResponseChunks(messages, { signal: controller.signal })
          .next();
    const assertion = expect(result).rejects.toBe(reason);
    await entered.promise;
    controller.abort(reason);
    await assertion;
    pending.resolve(new AIMessage('late'));
  }
);

it('sanitizes streaming errors and failing cleanup without losing the original safe code', async () => {
  const { model } = fixture({
    stream: () => ({
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          throw new HostModelServiceError('blocked');
        },
        return: async () => {
          throw new Error('PRIVATE cleanup');
        },
      }),
    }),
  });
  await expect(
    model._streamResponseChunks(messages, {}).next()
  ).rejects.toMatchObject({
    code: 'blocked',
    message: 'Host model service: blocked',
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
});

it.each([
  { ...descriptor, version: 2 },
  { ...descriptor, reference: '' },
  { ...descriptor, model: '  ' },
  { ...descriptor, streaming: undefined },
  { ...descriptor, tools: [tool, tool] },
])('rejects an incompatible descriptor before execution', (invalid) => {
  expect(() => fixture({}, invalid as HostModelDescriptor)).toThrow(
    HostModelServiceError
  );
});

it('uses host-approved tools through the native SDK tool lifecycle', async () => {
  const native = new FakeChatModel({
    responses: ['calling echo', 'finished'],
    toolCalls: [
      {
        id: 'echo-1',
        name: 'echo',
        args: { text: 'hello' },
        type: 'tool_call',
      },
    ],
    sleep: 0,
  });
  const requests: HostModelRequest[] = [];
  const { service } = fixture({
    stream: async function* (request, signal) {
      requests.push(request);
      yield* await native.stream([...request.messages], { signal });
    },
  });
  const invocations: string[] = [];
  const bindings = createHostToolBindings({
    identity,
    tools: [{ reference: 'host-echo', definition: tool.function }],
    service: {
      version: 1,
      invoke: async (request) =>
        request.calls.map(({ call }) => {
          invocations.push(call.name);
          return {
            toolCallId: call.id,
            status: 'success' as const,
            content: 'host result',
          };
        }),
    },
  });
  class HostProvider extends HostChatModel {
    constructor() {
      super({ identity, descriptor, service });
    }
  }
  const dispose = registerProvider({
    provider: 'host-model-tools-test',
    model: HostProvider,
    family: 'openai',
  });
  try {
    const run = await Run.create({
      runId: identity.runId,
      returnContent: true,
      tokenCounter: () => 1,
      customHandlers: bindings.customHandlers,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: 'host-model-tools-test' as RuntimeProviderName },
        toolDefinitions: bindings.toolDefinitions,
      },
    });
    const result = await run.processStream(
      { messages },
      { version: 'v2', configurable: { thread_id: 'thread' } }
    );
    expect(JSON.stringify(result)).toContain('finished');
    expect(invocations).toEqual(['echo']);
    expect(requests).toHaveLength(2);
    expect(
      requests.every((request) => request.toolNames.join() === 'echo')
    ).toBe(true);
    expect(
      requests[1].messages.some(
        (message) =>
          message.getType() === 'tool' && message.text === 'host result'
      )
    ).toBe(true);
  } finally {
    dispose();
  }
});

it('honors invocation tool choice over the bound default', async () => {
  const { model, requests } = fixture();
  await model
    .bindTools([tool], { tool_choice: 'auto' })
    .invoke(messages, { tool_choice: 'none' });
  expect(requests[0].toolChoice).toEqual({ mode: 'none' });
});

it('propagates early termination through the public stream API', async () => {
  let hostSignal: AbortSignal | undefined;
  const cleanup = jest.fn();
  const { model } = fixture({
    stream: async function* (_request, signal) {
      hostSignal = signal;
      try {
        yield new AIMessageChunk('first');
        yield new AIMessageChunk('second');
      } finally {
        cleanup();
      }
    },
  });
  for await (const chunk of await model.stream(messages)) {
    expect(chunk.text).toBe('first');
    break;
  }
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(hostSignal?.aborted).toBe(true);
  expect(cleanup).toHaveBeenCalledTimes(1);
});

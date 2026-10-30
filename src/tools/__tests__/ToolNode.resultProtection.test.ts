import { z } from 'zod';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { RemoveMessage, AIMessage, HumanMessage, ToolMessage  } from '@langchain/core/messages';
import {
  GraphInterrupt,
  Command,
  Overwrite,
  StateGraph,
  MessagesAnnotation,
  MemorySaver,
  START,
  END,
  isInterrupted,
} from '@langchain/langgraph';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type { AddressInfo } from 'node:net';
import type {
  ToolExecuteBatchRequest,
  ToolExecuteResult,
  EventHandler,
  ProgrammaticCache,
  RunStepResumeState,
  IState,
} from '@/types';
import type {
  ToolResultProtection,
  ToolResultProtectionResult,
} from '@/protection/toolResult';
import type { ToolOutputReferenceState } from '@/tools/toolOutputReferences';
import type { SettledToolBatchResult } from '@/tools/toolBatchReplay';
import {
  createCloudflareProgrammaticToolCallingTool,
  createCloudflareBashProgrammaticToolCallingTool,
} from '@/tools/cloudflare/CloudflareProgrammaticToolCalling';
import { snapshotToolMessageRemoval, validateToolMessageHistory, getToolMessageRelease, ToolResultProtectionError   } from '@/protection/toolResult';
import { createLocalProgrammaticToolCallingTool } from '@/tools/local/LocalProgrammaticToolCalling';
import {
  createProgrammaticToolCallingTool,
  executeTools,
} from '@/tools/ProgrammaticToolCalling';
import { createBashProgrammaticToolCallingTool } from '@/tools/BashProgrammaticToolCalling';
import {
  TOOL_BATCH_REPLAY_KEY,
  getToolBatchReplayState,
} from '@/tools/toolBatchReplay';
import { messagesStateReducer, createRemoveAllMessage } from '@/messages/reducer';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import fixtures from '@/protection/__tests__/fixtures/a1.json';
import { stableStringify } from '@/tools/eagerEventExecution';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { Constants, GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

function approve(content: string): ToolResultProtectionResult {
  return {
    version: 1,
    ok: true,
    value: { content, replacements: 1, categories: [] },
  };
}
function policy(
  overrides: Partial<ToolResultProtection> = {}
): ToolResultProtection {
  return {
    version: 1,
    toolNames: ['lookup'],
    timeoutMs: 10000,
    maxAttemptBytes: 65536,
    maxBufferedBytes: 262144,
    classify: () => 'prose',
    inspect: ({ content }) =>
      approve(
        content
          .replace(/A1_SECRET_CANARY_A/g, '[CREDENTIAL_1]')
          .replace(/a1.alice@example.invalid/g, '[EMAIL_1]')
      ),
    ...overrides,
  };
}
function deferred<T>() {
  return Promise.withResolvers<T>();
}
function state(id = 'call-control') {
  return {
    messages: [
      new AIMessage({
        content: '',
        tool_calls: [
          { id, name: 'lookup', args: { count: 42 }, type: 'tool_call' },
        ],
      }),
    ],
  };
}
function direct(fn: () => Promise<unknown> | unknown) {
  return new DynamicStructuredTool({
    name: 'lookup',
    description: 'Allowed control.',
    schema: z.object({ count: z.number() }),
    func: async ({ count }) => {
      expect(count).toBe(42);
      return await fn();
    },
  });
}
function observer(
  events: string[],
  host?: (request: ToolExecuteBatchRequest) => Promise<void> | void
) {
  const handler = BaseCallbackHandler.fromMethods({
    handleCustomEvent: async (name: string, data: unknown): Promise<void> => {
      if (name === GraphEvents.ON_TOOL_EXECUTE && host != null)
        await host(data as ToolExecuteBatchRequest);
      if (name === GraphEvents.ON_RUN_STEP_COMPLETED)
        events.push(JSON.stringify(data));
    },
    handleToolEnd: (output: unknown): void => {
      events.push(JSON.stringify(output));
    },
    handleToolError: (error: Error): void => {
      events.push(error.message);
    },
  });
  handler.awaitHandlers = true;
  return handler;
}
function assertNoCanary(value: unknown) {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  for (const canary of fixtures.canaries) expect(text).not.toContain(canary);
}

afterEach(() => jest.restoreAllMocks());

it.each(
  fixtures.cases.filter(
    (entry) =>
      entry.target.source === 'tool_argument' && entry.expected.content != null
  )
)(
  'applies A1 $id before direct callbacks, references and reuse',
  async (entry) => {
    const events: string[] = [];
    let executions = 0;
    const error = entry.target.outcome === 'error';
    const tool = direct(() => {
      executions++;
      if (error) throw new Error(entry.chunks.join(''));
      return entry.chunks.join('');
    });
    const gate = policy({
      inspect: ({ content, target, toolName, toolCallId }) => {
        expect(content).toBe(entry.chunks.join(''));
        expect(target.outcome).toBe(error ? 'error' : 'success');
        expect(toolName).toBe('lookup');
        expect(toolCallId).toBe('call-control');
        return approve(entry.expected.content!);
      },
    });
    const refs = new ToolOutputReferenceRegistry();
    const node = new ToolNode({
      trace: true,
      tools: [tool],
      toolResultProtection: gate,
      toolOutputRegistry: refs,
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    const result = (await node.invoke(state(), {
      configurable: { run_id: 'a1-tool' },
      callbacks: [observer(events)],
    })) as { messages: ToolMessage[] };
    expect(executions).toBe(1);
    expect(result.messages[0].status).toBe(error ? 'error' : 'success');
    expect(result.messages[0].tool_call_id).toBe('call-control');
    expect(result.messages[0].content).toContain(entry.expected.content);
    expect(events.join('')).toContain(
      JSON.stringify(entry.expected.content).slice(1, -1)
    );
    assertNoCanary(events.join(''));
    assertNoCanary(result);
    if (!error)
      expect(refs.get('a1-tool', 'tool0turn0')).toBe(entry.expected.content);
  }
);

it.each(['direct', 'host'] as const)(
  'awaits required policy before %s completion/storage',
  async (path) => {
    const started = deferred<void>();
    const decision = deferred<ToolResultProtectionResult>();
    const events: string[] = [];
    const refs = new ToolOutputReferenceRegistry();
    let executions = 0;
    const tool = direct(() => {
      executions++;
      return fixtures.canaries[0];
    });
    const node = new ToolNode({
      trace: true,
      tools: [tool],
      eventDrivenMode: path === 'host',
      toolResultProtection: policy({
        inspect: () => {
          started.resolve();
          return decision.promise;
        },
      }),
      toolOutputRegistry: refs,
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    const task = node.invoke(state(), {
      configurable: { run_id: 'await-tool' },
      callbacks: [
        observer(events, (request) => {
          executions++;
          request.onResult?.({
            toolCallId: request.toolCalls[0].id,
            status: 'success',
            content: fixtures.canaries[0],
          });
          request.resolve([
            {
              toolCallId: request.toolCalls[0].id,
              status: 'success',
              content: fixtures.canaries[0],
            },
          ]);
        }),
      ],
    });
    await started.promise;
    expect(events).toEqual([]);
    expect(refs.get('await-tool', 'tool0turn0')).toBeUndefined();
    decision.resolve(approve('Allowed release control'));
    const result = await task;
    expect(executions).toBe(1);
    assertNoCanary(result);
    assertNoCanary(events.join(''));
    expect(events.join('')).toContain('Allowed release control');
    expect(refs.get('await-tool', 'tool0turn0')).toBe(
      'Allowed release control'
    );
  }
);

it.each(['direct', 'host'] as const)(
  'protects %s post-hook replacement before completion/reference registration',
  async (path) => {
    const hooks = new HookRegistry();
    hooks.register('PostToolUse', {
      hooks: [async () => ({ updatedOutput: fixtures.canaries[0] })],
    });
    const inspected: string[] = [];
    const events: string[] = [];
    const refs = new ToolOutputReferenceRegistry();
    const gate = policy({
      inspect: ({ content }) => {
        inspected.push(content);
        return approve(
          content === fixtures.canaries[0] ? '[EMAIL_1]' : content
        );
      },
    });
    const node = new ToolNode({
      trace: true,
      tools: [direct(() => 'Initial allowed control')],
      eventDrivenMode: path === 'host',
      hookRegistry: hooks,
      toolResultProtection: gate,
      toolOutputRegistry: refs,
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    const result = await node.invoke(state(), {
      configurable: { run_id: 'hook-tool' },
      callbacks: [
        observer(events, (request) =>
          request.resolve([
            {
              toolCallId: 'call-control',
              status: 'success',
              content: 'Initial allowed control',
            },
          ])
        ),
      ],
    });
    expect(inspected).toEqual([
      'Initial allowed control',
      fixtures.canaries[0],
    ]);
    expect(refs.get('hook-tool', 'tool0turn0')).toBe('[EMAIL_1]');
    assertNoCanary(events.join(''));
    assertNoCanary(result);
  }
);

it('does not substitute an optional throwing PostToolUse hook for required inspection', async () => {
  const hooks = new HookRegistry();
  hooks.register('PostToolUse', {
    hooks: [
      async () => {
        throw new Error('Optional observer failed');
      },
    ],
  });
  const node = new ToolNode({
    trace: true,
    tools: [direct(() => fixtures.canaries[0])],
    hookRegistry: hooks,
    toolResultProtection: policy(),
  });
  const result = await node.invoke(state());
  assertNoCanary(result);
  expect(JSON.stringify(result)).toContain('[EMAIL_1]');
});

it.each([
  'missing',
  'throw',
  'reject',
  'version',
  'result-version',
  'overflow',
  'unsupported',
] as const)('fails closed for required policy %s', async (kind) => {
  const gate = policy();
  let executions = 0;
  const events: string[] = [];
  if (kind === 'missing')
    Object.defineProperty(gate, 'inspect', { value: undefined });
  if (kind === 'version') Object.defineProperty(gate, 'version', { value: 2 });
  if (kind === 'throw')
    Object.defineProperty(gate, 'inspect', {
      value: () => {
        throw new Error(fixtures.canaries[0]);
      },
    });
  if (kind === 'reject')
    Object.defineProperty(gate, 'inspect', {
      value: () => Promise.reject(new Error(fixtures.canaries[0])),
    });
  if (kind === 'result-version')
    Object.defineProperty(gate, 'inspect', {
      value: () => ({
        version: 2,
        ok: true,
        value: {
          content: fixtures.canaries[0],
          replacements: 0,
          categories: [],
        },
      }),
    });
  if (kind === 'overflow')
    Object.defineProperty(gate, 'maxAttemptBytes', { value: 512 });
  if (kind === 'unsupported')
    Object.defineProperty(gate, 'classify', { value: () => 'unsupported' });
  const outcome = (async () => {
    const node = new ToolNode({
      trace: true,
      tools: [
        direct(() => {
          executions++;
          return fixtures.canaries[0];
        }),
      ],
      toolResultProtection: gate,
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    return node.invoke(state(), { callbacks: [observer(events)] });
  })().catch((error: Error) => error);
  const error = await outcome;
  expect(error).toBeInstanceOf(Error);
  assertNoCanary(String(error));
  assertNoCanary(events.join(''));
  expect(events.join('')).not.toContain(fixtures.canaries[0]);
  if (kind === 'missing' || kind === 'version') expect(executions).toBe(0);
});

it.each(['stop', 'timeout'] as const)(
  'quarantines %s and late required policy completion',
  async (kind) => {
    const controller = new AbortController();
    const started = deferred<void>();
    const decision = deferred<ToolResultProtectionResult>();
    const events: string[] = [];
    const node = new ToolNode({
      trace: true,
      tools: [direct(() => fixtures.canaries[0])],
      toolResultProtection: policy({
        timeoutMs: kind === 'timeout' ? 30 : 10000,
        inspect: () => {
          started.resolve();
          return decision.promise;
        },
      }),
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    const task = node.invoke(state(), {
      signal: controller.signal,
      callbacks: [observer(events)],
    });
    const outcome = task.catch((error: Error) => error);
    await started.promise;
    if (kind === 'stop') controller.abort();
    expect(await outcome).toBeInstanceOf(Error);
    decision.resolve(approve('Late forbidden control'));
    await new Promise((resolve) => setImmediate(resolve));
    expect(events.join('')).not.toContain('Late forbidden control');
    assertNoCanary(events.join(''));
  }
);

it.each([
  'tuple artifact',
  'ToolMessage artifact',
  'structured',
  'JSON',
  'alias',
  'injected',
] as const)(
  'rejects uncertified %s instead of rewriting public payloads',
  async (kind) => {
    const raw = fixtures.canaries[0];
    const events: string[] = [];
    if (kind === 'alias' || kind === 'injected') {
      const node = new ToolNode({
        trace: true,
        tools: [direct(() => 'unused')],
        eventDrivenMode: true,
        toolResultProtection: policy(),
        toolCallStepIds: new Map([['call-control', 'step-control']]),
      });
      const result = {
        toolCallId: 'call-control',
        status: 'success' as const,
        content: 'Allowed control',
        ...(kind === 'alias'
          ? { raw_output: raw }
          : { injectedMessages: [{ role: 'user' as const, content: raw }] }),
      };
      await expect(
        node.invoke(state(), {
          callbacks: [observer(events, (request) => request.resolve([result]))],
        })
      ).rejects.toBeInstanceOf(ToolResultProtectionError);
    } else {
      let output: unknown = JSON.stringify({ email: raw });
      if (kind === 'tuple artifact') output = ['Allowed control', { raw }];
      if (kind === 'ToolMessage artifact')
        output = new ToolMessage({
          content: 'Allowed control',
          tool_call_id: 'call-control',
          artifact: { raw },
        });
      if (kind === 'structured') output = { content: raw };
      const tool = direct(() => output);
      if (kind === 'tuple artifact')
        tool.responseFormat = 'content_and_artifact';
      const node = new ToolNode({
        trace: true,
        tools: [tool],
        toolResultProtection: policy({
          classify: (content) =>
            content.startsWith('{') ? 'unsupported' : 'prose',
        }),
        toolCallStepIds: new Map([['call-control', 'step-control']]),
      });
      await expect(
        node.invoke(state(), { callbacks: [observer(events)] })
      ).rejects.toBeInstanceOf(ToolResultProtectionError);
    }
    assertNoCanary(events.join(''));
  }
);

it.each([false, true])(
  'keeps absence/unselected tool behavior unchanged (configured=%s)',
  async (configured) => {
    const gate = configured
      ? policy({
        toolNames: ['another-tool'],
        inspect: () => {
          throw new Error('Must not run');
        },
      })
      : undefined;
    const node = new ToolNode({
      trace: true,
      tools: [direct(() => fixtures.canaries[0])],
      toolResultProtection: gate,
    });
    const result = await node.invoke(state());
    expect(JSON.stringify(result)).toContain(fixtures.canaries[0]);
  }
);

it.each([false, true])(
  'uses protected real host/eager results exactly once in subsequent model state (eager=%s)',
  async (eager) => {
    let executions = 0;
    const completions: string[] = [];
    const wireInputs: string[] = [];
    const inspected: string[] = [];
    class Model extends FakeChatModel {
      override async *_streamResponseChunks(
        ...args: Parameters<FakeChatModel['_streamResponseChunks']>
      ): ReturnType<FakeChatModel['_streamResponseChunks']> {
        wireInputs.push(JSON.stringify(args[0]));
        yield* super._streamResponseChunks(...args);
      }
    }
    const handlers: Record<string, EventHandler> = {
      [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: async (_event, data): Promise<void> => {
          const request = data as ToolExecuteBatchRequest;
          executions++;
          expect(request.toolCalls[0].args).toEqual({ count: 42 });
          request.onResult?.({
            toolCallId: request.toolCalls[0].id,
            status: 'success',
            content: fixtures.canaries[0],
          });
          request.resolve([
            {
              toolCallId: request.toolCalls[0].id,
              status: 'success',
              content: fixtures.canaries[0],
            },
          ]);
        },
      },
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (_event, data): void => {
          completions.push(JSON.stringify(data));
        },
      },
    };
    const run = await Run.create({
      runId: `c1-host-${eager}`,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
        instructions: 'Allowed instructions.',
        toolDefinitions: [
          {
            name: 'lookup',
            description: 'Allowed control.',
            parameters: {
              type: 'object',
              properties: { count: { type: 'number' } },
            },
          },
        ],
      },
      customHandlers: handlers,
      toolResultProtection: policy({
        inspect: ({ content }) => {
          inspected.push(content);
          return approve('[EMAIL_1]');
        },
      }),
      toolOutputReferences: { enabled: !eager },
      eagerEventToolExecution: { enabled: eager },
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph!.overrideModel = new Model({
      responses: ['', 'Final allowed control'],
      toolCalls: [
        {
          id: 'call-control',
          name: 'lookup',
          args: { count: 42 },
          type: 'tool_call',
        },
      ],
    });
    await run.processStream(
      { messages: [new HumanMessage('Allowed control')] },
      { version: 'v2', configurable: { thread_id: `c1-host-${eager}` } }
    );
    expect(executions).toBe(1);
    expect(inspected).toEqual([fixtures.canaries[0]]);
    expect(wireInputs).toHaveLength(2);
    expect(wireInputs[1]).toContain('[EMAIL_1]');
    assertNoCanary(wireInputs[1]);
    assertNoCanary(completions.join(''));
    expect(completions.join('')).toContain('[EMAIL_1]');
    expect(JSON.stringify(run.Graph!.getRunMessages())).toContain(
      'Final allowed control'
    );
  }
);

it('protects host error text before hooks, completion and ToolMessage reuse', async () => {
  const events: string[] = [];
  const node = new ToolNode({
    trace: true,
    tools: [direct(() => 'unused')],
    eventDrivenMode: true,
    toolResultProtection: policy(),
    toolCallStepIds: new Map([['call-control', 'step-control']]),
  });
  const result = (await node.invoke(state(), {
    callbacks: [
      observer(events, (request) =>
        request.resolve([
          {
            toolCallId: 'call-control',
            content: fixtures.canaries[1],
            status: 'error',
            errorMessage: fixtures.canaries[0],
          },
        ])
      ),
    ],
  })) as { messages: ToolMessage[] };
  expect(result.messages[0].status).toBe('error');
  expect(result.messages[0].content).toContain('[EMAIL_1]');
  assertNoCanary(result);
  assertNoCanary(events.join(''));
});

it('retains a shared result lease after timeout until an ignoring handler settles', async () => {
  const decision = deferred<ToolResultProtectionResult>();
  const gate = policy({
    timeoutMs: 30,
    maxAttemptBytes: 4096,
    maxBufferedBytes: 4096,
    inspect: () => decision.promise,
  });
  const node = new ToolNode({
    trace: true,
    tools: [direct(() => 'x'.repeat(500))],
    toolResultProtection: gate,
  });
  await expect(node.invoke(state('first'))).rejects.toMatchObject({
    code: 'timeout',
  });
  await expect(node.invoke(state('second'))).rejects.toMatchObject({
    code: 'overflow',
  });
  decision.resolve(approve('Allowed retry control'));
  await new Promise((resolve) => setImmediate(resolve));
  expect(JSON.stringify(await node.invoke(state('retry')))).toContain(
    'Allowed retry control'
  );
});

it.each(['classify', 'inspect'])(
  'fails closed when synchronous %s exceeds the release deadline',
  async (kind) => {
    const busy = (): void => {
      const end = performance.now() + 30;
      while (performance.now() < end) {
        /* bounded test */
      }
    };
    const gate = policy({
      timeoutMs: 10,
      classify: () => {
        if (kind === 'classify') busy();
        return 'prose';
      },
      inspect: () => {
        if (kind === 'inspect') busy();
        return approve('Late approved control');
      },
    });
    const events: string[] = [];
    const node = new ToolNode({
      trace: true,
      tools: [direct(() => fixtures.canaries[0])],
      toolResultProtection: gate,
    });
    await expect(
      node.invoke(state(), { callbacks: [observer(events)] })
    ).rejects.toMatchObject({ code: 'timeout' });
    expect(events.join('')).not.toContain('Late approved control');
    assertNoCanary(events.join(''));
  }
);

it('keeps concurrent attempts and repeated ids isolated with one immutable policy', async () => {
  const a = deferred<ToolResultProtectionResult>();
  const b = deferred<ToolResultProtectionResult>();
  const both = deferred<void>();
  const seen: string[] = [];
  const gate = policy({
    inspect: ({ content }) => {
      seen.push(content);
      if (seen.length === 2) both.resolve();
      return content === fixtures.canaries[0] ? a.promise : b.promise;
    },
  });
  const nodeA = new ToolNode({
    tools: [direct(() => fixtures.canaries[0])],
    toolResultProtection: gate,
  });
  const nodeB = new ToolNode({
    tools: [direct(() => fixtures.canaries[1])],
    toolResultProtection: gate,
  });
  const taskA = nodeA.invoke(state());
  const taskB = nodeB.invoke(state());
  await both.promise;
  a.resolve(approve('Tenant A allowed control'));
  b.resolve(approve('Tenant B allowed control'));
  expect(JSON.stringify(await taskA)).toContain('Tenant A allowed control');
  expect(JSON.stringify(await taskB)).toContain('Tenant B allowed control');
});

it('protects selected nested programmatic tools and preserves success/error status', async () => {
  const result = await executeTools(
    [{ id: 'inner-control', name: 'lookup', input: { count: 42 } }],
    new Map([['lookup', direct(() => fixtures.canaries[0])]]),
    'run_tools_with_code',
    { policy: policy() }
  );
  expect(result).toEqual([
    { call_id: 'inner-control', result: '[EMAIL_1]', is_error: false },
  ]);
  const error = await executeTools(
    [{ id: 'inner-control', name: 'lookup', input: { count: 42 } }],
    new Map([
      [
        'lookup',
        direct(() => {
          throw new Error(fixtures.canaries[0]);
        }),
      ],
    ]),
    'run_tools_with_code',
    { policy: policy() }
  );
  expect(error).toEqual([
    {
      call_id: 'inner-control',
      result: null,
      is_error: true,
      error_message: '[EMAIL_1]',
    },
  ]);
});

it('runs the real local bash tool bridge with only canonical selected output', async () => {
  let executions = 0;
  const lookup = direct(() => {
    executions++;
    return fixtures.canaries[0];
  });
  const runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
  const result = await runner.invoke(
    {
      lang: 'bash',
      code: 'lookup \'{"count":42}\'',
      tool_manifest: ['lookup'],
    },
    {
      toolCall: {
        id: 'local-control',
        name: 'run_tools_with_code',
        type: 'tool_call',
        args: {},
        toolMap: new Map([['lookup', lookup]]),
        toolDefs: [
          {
            name: 'lookup',
            description: 'Allowed control.',
            allowed_callers: ['code_execution'],
            parameters: {
              type: 'object',
              properties: { count: { type: 'number' } },
            },
          },
        ],
        toolResultProtection: policy(),
      },
    }
  );
  expect(executions).toBe(1);
  expect(JSON.stringify(result)).toContain('[EMAIL_1]');
  assertNoCanary(result);
});

it('fails a blocked real local bridge call without returning the raw result', async () => {
  const runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
  await expect(
    runner.invoke(
      {
        lang: 'bash',
        code: 'lookup \'{"count":42}\'',
        tool_manifest: ['lookup'],
      },
      {
        toolCall: {
          id: 'local-control',
          name: 'run_tools_with_code',
          type: 'tool_call',
          args: {},
          toolMap: new Map([['lookup', direct(() => fixtures.canaries[0])]]),
          toolDefs: [
            {
              name: 'lookup',
              allowed_callers: ['code_execution'],
              parameters: {
                type: 'object',
                properties: { count: { type: 'number' } },
              },
            },
          ],
          toolResultProtection: policy({
            inspect: () => ({
              version: 1,
              ok: false,
              error: { code: 'blocked' },
            }),
          }),
        },
      }
    )
  ).rejects.toMatchObject({ code: 'blocked' });
});

it.each(['duplicate', 'missing', 'rejection'])(
  'fails closed on a selected host %s envelope',
  async (kind) => {
    const events: string[] = [];
    const node = new ToolNode({
      trace: true,
      tools: [direct(() => 'unused')],
      eventDrivenMode: true,
      toolResultProtection: policy(),
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    await expect(
      node.invoke(state(), {
        callbacks: [
          observer(events, (request) => {
            if (kind === 'rejection') {
              request.reject(new Error(fixtures.canaries[0]));
              return;
            }
            const result = {
              toolCallId: 'call-control',
              content: fixtures.canaries[0],
              status: 'success' as const,
            };
            request.resolve(kind === 'missing' ? [] : [result, result]);
          }),
        ],
      })
    ).rejects.toBeInstanceOf(Error);
    assertNoCanary(events.join(''));
  }
);

it('inherits C1 policy into foreground child tools before parent model reuse', async () => {
  let executions = 0;
  const inspected: string[] = [];
  const lookup = direct(() => {
    executions++;
    return fixtures.canaries[0];
  });
  const run = await Run.create({
    runId: 'c1-child',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'parent',
          provider: Providers.OPENAI,
          instructions: 'Allowed parent instructions.',
          subagentConfigs: [
            {
              type: 'worker',
              name: 'Worker',
              description: 'Allowed child.',
              agentInputs: {
                agentId: 'child',
                provider: Providers.OPENAI,
                instructions: 'Allowed child instructions.',
                tools: [lookup],
              },
            },
          ],
        },
      ],
    },
    toolResultProtection: policy({
      inspect: ({ content }) => {
        inspected.push(content);
        return approve('[EMAIL_1]');
      },
    }),
    returnContent: true,
    skipCleanup: true,
  });
  run.Graph!.overrideModel = new FakeChatModel({
    responses: ['', 'Parent allowed answer'],
    toolCalls: [
      {
        id: 'child-control',
        name: Constants.SUBAGENT,
        args: { description: 'Allowed task.', subagent_type: 'worker' },
        type: 'tool_call',
      },
    ],
  });
  run.Graph!.setSubagentModelOverride(
    new FakeChatModel({
      responses: ['', 'Child allowed answer'],
      toolCalls: [
        {
          id: 'lookup-control',
          name: 'lookup',
          args: { count: 42 },
          type: 'tool_call',
        },
      ],
    })
  );
  await run.processStream(
    { messages: [new HumanMessage('Allowed control')] },
    { version: 'v2', configurable: { thread_id: 'c1-child' } }
  );
  expect(executions).toBe(1);
  expect(inspected).toEqual([fixtures.canaries[0]]);
  assertNoCanary(JSON.stringify(run.Graph!.getRunMessages()));
  expect(JSON.stringify(run.Graph!.getRunMessages())).toContain(
    'Parent allowed answer'
  );
});

it.each(['direct', 'host'] as const)(
  'keeps denied %s authority and execute-once accounting intact',
  async (path) => {
    let executed = 0;
    const events: string[] = [];
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async () => ({ decision: 'deny', reason: 'Allowed denial control' }),
      ],
    });
    const node = new ToolNode({
      trace: true,
      tools: [
        direct(() => {
          executed++;
          return 'Must not execute';
        }),
      ],
      eventDrivenMode: path === 'host',
      hookRegistry: hooks,
      toolResultProtection: policy(),
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    const result = (await node.invoke(state(), {
      callbacks: [
        observer(events, () => {
          executed++;
        }),
      ],
    })) as { messages: ToolMessage[] };
    expect(executed).toBe(0);
    expect(result.messages[0].status).toBe('error');
    expect(result.messages[0].tool_call_id).toBe('call-control');
    expect(result.messages[0].content).toContain('Allowed denial control');
  }
);

it.each([true, false])(
  'preserves SDK-owned safety interruption identity with tool handling=%s',
  async (handleToolErrors) => {
    const safety = new PreparedSubagentError(
      'Allowed prepared-execution safety control'
    );
    const node = new ToolNode({
      tools: [
        direct(() => {
          throw safety;
        }),
      ],
      toolResultProtection: policy(),
      handleToolErrors,
    });
    await expect(node.invoke(state())).rejects.toBe(safety);
  }
);

it.each([true, false])(
  'preserves GraphInterrupt approval/resume identity with tool handling=%s',
  async (handleToolErrors) => {
    const safety = new GraphInterrupt([]);
    const node = new ToolNode({
      tools: [
        direct(() => {
          throw safety;
        }),
      ],
      toolResultProtection: policy(),
      handleToolErrors,
    });
    await expect(node.invoke(state())).rejects.toMatchObject({
      name: 'GraphInterrupt',
      interrupts: [],
    });
  }
);

it.each(['rejection', 'missing', 'duplicate'])(
  'fails a real selected eager %s before any completion',
  async (kind) => {
    const completions: string[] = [];
    const handlers: Record<string, EventHandler> = {
      [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, data): void => {
          const request = data as ToolExecuteBatchRequest;
          if (kind === 'rejection') {
            request.reject(new Error(fixtures.canaries[0]));
            return;
          }
          const result = {
            toolCallId: request.toolCalls[0].id,
            status: 'success' as const,
            content: fixtures.canaries[0],
          };
          request.resolve(kind === 'missing' ? [] : [result, result]);
        },
      },
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (_event, data): void => {
          completions.push(JSON.stringify(data));
        },
      },
    };
    const run = await Run.create({
      runId: `c1-eager-${kind}`,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
        instructions: 'Allowed control.',
        toolDefinitions: [
          {
            name: 'lookup',
            parameters: {
              type: 'object',
              properties: { count: { type: 'number' } },
            },
          },
        ],
      },
      customHandlers: handlers,
      eagerEventToolExecution: { enabled: true },
      toolResultProtection: policy(),
      skipCleanup: true,
    });
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['', 'Must not continue'],
      toolCalls: [
        {
          id: 'call-control',
          name: 'lookup',
          args: { count: 42 },
          type: 'tool_call',
        },
      ],
    });
    const error = await run
      .processStream(
        { messages: [new HumanMessage('Allowed control')] },
        { version: 'v2', configurable: { thread_id: `c1-eager-${kind}` } }
      )
      .catch((value: Error) => value);
    expect(error).toBeInstanceOf(Error);
    assertNoCanary(String(error));
    expect(completions).toEqual([]);
    assertNoCanary(run.Graph!.getRunMessages());
  }
);

it('protects nested diagnostics raised before the native _call', async () => {
  const bad = new DynamicStructuredTool({
    name: 'lookup',
    description: 'Allowed schema.',
    schema: z.object({
      count: z.number().refine(() => false, { message: fixtures.canaries[0] }),
    }),
    func: async () => {
      throw new Error('Schema must prevent execution');
    },
  });
  const result = await executeTools(
    [{ id: 'schema-control', name: 'lookup', input: { count: 42 } }],
    new Map([['lookup', bad]]),
    'run_tools_with_code',
    { policy: policy() }
  );
  expect(result[0].is_error).toBe(true);
  expect(result[0].error_message).toContain('[EMAIL_1]');
  assertNoCanary(result);
});

it('protects a denied local bridge response without authorizing the tool', async () => {
  let executions = 0;
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', {
    hooks: [async () => ({ decision: 'deny', reason: fixtures.canaries[0] })],
  });
  const runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
  const result = await runner.invoke(
    {
      lang: 'bash',
      code: 'lookup \'{"count":42}\' || true',
      tool_manifest: ['lookup'],
    },
    {
      toolCall: {
        id: 'local-denial',
        name: 'run_tools_with_code',
        type: 'tool_call',
        args: {},
        toolMap: new Map([
          [
            'lookup',
            direct(() => {
              executions++;
              return 'Must not run';
            }),
          ],
        ]),
        toolDefs: [
          {
            name: 'lookup',
            allowed_callers: ['code_execution'],
            parameters: {
              type: 'object',
              properties: { count: { type: 'number' } },
            },
          },
        ],
        toolResultProtection: policy(),
        hookContext: { registry: hooks, runId: 'local-denial' },
      },
    }
  );
  expect(executions).toBe(0);
  assertNoCanary(result);
  expect(JSON.stringify(result)).toContain('[EMAIL_1]');
});

it.each(['python', 'bash'] as const)(
  'keeps required nested failures terminal through the remote %s runner',
  async (runtime) => {
    const requests: string[] = [];
    const server = createServer((request, response) => {
      let body = '';
      request.on('data', (chunk: Buffer): void => {
        body += chunk.toString();
      });
      request.on('end', (): void => {
        requests.push(body);
        response.setHeader('Content-Type', 'application/json');
        response.end(
          JSON.stringify(
            requests.length === 1
              ? {
                status: 'tool_call_required',
                continuation_token: 'allowed-continuation',
                tool_calls: [
                  {
                    id: 'inner-control',
                    name: 'lookup',
                    input: { count: 42 },
                  },
                ],
              }
              : { status: 'completed', stdout: 'Allowed control', files: [] }
          )
        );
      });
    });
    await new Promise<void>((resolve) =>
      server.listen(0, '127.0.0.1', resolve)
    );
    try {
      const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
      const runner =
        runtime === 'python'
          ? createProgrammaticToolCallingTool({ baseUrl })
          : createBashProgrammaticToolCallingTool({ baseUrl });
      await expect(
        runner.invoke(
          {
            code:
              runtime === 'python'
                ? 'print(await lookup(count=42))'
                : 'lookup \'{"count":42}\'',
            tool_manifest: ['lookup'],
          },
          {
            toolCall: {
              id: 'remote-control',
              name: runner.name,
              type: 'tool_call',
              args: {},
              toolMap: new Map([
                ['lookup', direct(() => fixtures.canaries[0])],
              ]),
              toolDefs: [
                {
                  name: 'lookup',
                  allowed_callers: ['code_execution'],
                  parameters: {
                    type: 'object',
                    properties: { count: { type: 'number' } },
                  },
                },
              ],
              toolResultProtection: policy({
                inspect: () => ({
                  version: 1,
                  ok: false,
                  error: { code: 'blocked' },
                }),
              }),
            },
          }
        )
      ).rejects.toMatchObject({ code: 'blocked' });
      expect(requests).toHaveLength(1);
      assertNoCanary(requests);
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  }
);

it.each(['direct', 'host'] as const)(
  'protects approved %s resume results without repeating execution',
  async (path) => {
    let executions = 0;
    const inspected: string[] = [];
    const events: string[] = [];
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async () => ({ decision: 'ask', reason: 'Allowed approval control' }),
      ],
    });
    const node = new ToolNode({
      trace: true,
      tools: [
        direct(() => {
          executions++;
          return fixtures.canaries[0];
        }),
      ],
      eventDrivenMode: path === 'host',
      hookRegistry: hooks,
      humanInTheLoop: { enabled: true },
      toolResultProtection: policy({
        inspect: ({ content }) => {
          inspected.push(content);
          return approve('[EMAIL_1]');
        },
      }),
      toolOutputReferences: { enabled: true },
      toolCallStepIds: new Map([['call-control', 'step-control']]),
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('tools', node)
      .addEdge(START, 'tools')
      .addEdge('tools', END)
      .compile({ checkpointer: new MemorySaver() });
    const config = {
      configurable: {
        thread_id: `c1-resume-${path}`,
        run_id: `c1-resume-${path}`,
      },
      callbacks: [
        observer(events, (request) => {
          executions++;
          request.resolve([
            {
              toolCallId: 'call-control',
              status: 'success',
              content: fixtures.canaries[0],
            },
          ]);
        }),
      ],
    };
    const interrupted = await graph.invoke(state(), config);
    expect(isInterrupted(interrupted)).toBe(true);
    expect(executions).toBe(0);
    expect(inspected).toEqual([]);
    const resumed = await graph.invoke(
      new Command({ resume: [{ type: 'approve' }] }),
      config
    );
    expect(executions).toBe(1);
    expect(inspected).toEqual([fixtures.canaries[0]]);
    assertNoCanary(resumed);
    assertNoCanary(events.join(''));
    expect(JSON.stringify(resumed)).toContain('[EMAIL_1]');
  }
);

it('keeps native callback child APIs usable without pre-release raw observations', async () => {
  const events: string[] = [];
  const lookup = new DynamicStructuredTool({
    name: 'lookup',
    description: 'Allowed child callback control.',
    schema: z.object({ count: z.number() }),
    func: async (_input, manager) => {
      expect(manager).toBeDefined();
      expect(manager!.getChild()).toBeDefined();
      await manager!.handleText(fixtures.canaries[0]);
      return fixtures.canaries[0];
    },
  });
  const node = new ToolNode({
    trace: true,
    tools: [lookup],
    toolResultProtection: policy(),
    toolCallStepIds: new Map([['call-control', 'step-control']]),
  });
  const callbacks = observer(events);
  callbacks.handleText = (text: string): void => {
    events.push(text);
  };
  const result = await node.invoke(state(), { callbacks: [callbacks] });
  assertNoCanary(events.join(''));
  assertNoCanary(result);
  expect(events.join('')).toContain('[EMAIL_1]');
});

it('rejects an opaque producer message-id alias rather than rewriting control identity', async () => {
  const events: string[] = [];
  const lookup = direct(
    () =>
      new ToolMessage({
        id: fixtures.canaries[0],
        tool_call_id: 'call-control',
        content: 'Allowed control',
      })
  );
  const node = new ToolNode({
    trace: true,
    tools: [lookup],
    toolResultProtection: policy(),
  });
  await expect(
    node.invoke(state(), { callbacks: [observer(events)] })
  ).rejects.toMatchObject({ code: 'unsupported' });
  assertNoCanary(events.join(''));
});

it('preserves native error callbacks and canonical error-handler ownership', async () => {
  let ends = 0;
  const errors: string[] = [];
  const handled: string[] = [];
  const callback = BaseCallbackHandler.fromMethods({
    handleToolEnd: (): void => {
      ends++;
    },
    handleToolError: (error: Error): void => {
      errors.push(error.message);
    },
  });
  callback.awaitHandlers = true;
  const node = new ToolNode({
    trace: true,
    tools: [
      direct(() => {
        throw new Error(fixtures.canaries[0]);
      }),
    ],
    toolResultProtection: policy(),
    errorHandler: async (data) => {
      handled.push(data.error!.message);
      return false;
    },
  });
  const result = await node.invoke(state(), { callbacks: [callback] });
  expect(ends).toBe(0);
  expect(errors).toEqual(['[EMAIL_1]']);
  expect(handled).toEqual(['[EMAIL_1]']);
  assertNoCanary(result);
});

it('rejects selected accessor output before native callbacks can observe its exception', async () => {
  const events: string[] = [];
  const message = new ToolMessage({
    tool_call_id: 'call-control',
    content: 'Allowed control',
  });
  Object.defineProperty(message, 'content', {
    get: () => {
      throw new Error(fixtures.canaries[0]);
    },
  });
  const node = new ToolNode({
    trace: true,
    tools: [direct(() => message)],
    toolResultProtection: policy(),
  });
  await expect(
    node.invoke(state(), { callbacks: [observer(events)] })
  ).rejects.toMatchObject({ code: 'unavailable' });
  assertNoCanary(events.join(''));
});

it('retains full approved reference content while projecting only the model preview', async () => {
  const raw = 'Allowed control '.repeat(150) + fixtures.canaries[0];
  const canonical = raw.replace(fixtures.canaries[0], '[EMAIL_1]');
  const inspected: string[] = [];
  const piped: string[] = [];
  const pipe = new DynamicStructuredTool({
    name: 'pipe',
    description: 'Allowed pipe control.',
    schema: z.object({ command: z.string() }),
    func: async ({ command }) => {
      piped.push(command);
      return 'Allowed piped control';
    },
  });
  const node = new ToolNode({
    tools: [direct(() => raw), pipe],
    maxToolResultChars: 128,
    toolOutputReferences: { enabled: true, maxOutputSize: 8192 },
    toolResultProtection: policy({
      inspect: ({ content }) => {
        inspected.push(content);
        return approve(content.replace(fixtures.canaries[0], '[EMAIL_1]'));
      },
    }),
  });
  const config = { configurable: { run_id: 'full-ref-control' } };
  const result = (await node.invoke(state(), config)) as {
    messages: ToolMessage[];
  };
  expect((result.messages[0].content as string).length).toBeLessThan(
    canonical.length
  );
  expect(
    node._unsafeGetToolOutputRegistry()!.get('full-ref-control', 'tool0turn0')
  ).toBe(canonical);
  expect(inspected).toEqual([raw]);
  await node.invoke(
    {
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'pipe-control',
              name: 'pipe',
              args: {
                command: '{{' + 'tool0turn0' + '}}',
              },
            },
          ],
        }),
      ],
    },
    config
  );
  expect(piped).toEqual([canonical]);
  assertNoCanary(result);
});

it('re-inspects checkpoint-owned references after mixed approval replay without repeating side effects', async () => {
  let lookups = 0;
  let approvals = 0;
  const inspected: string[] = [];
  const raw = 'Allowed replay control '.repeat(80) + fixtures.canaries[0];
  const canonical = raw.replace(fixtures.canaries[0], '[EMAIL_1]');
  const lookup = direct(() => {
    lookups++;
    return raw;
  });
  const approval = new DynamicStructuredTool({
    name: 'approval_control',
    description: 'Allowed approval.',
    schema: z.object({ count: z.number() }),
    func: async () => {
      approvals++;
      return 'Approved control';
    },
  });
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', {
    hooks: [
      async (input) => ({
        decision: input.toolName === 'approval_control' ? 'ask' : 'allow',
        reason: 'Allowed approval control',
      }),
    ],
  });
  const gate = policy({
    inspect: ({ content }) => {
      inspected.push(content);
      return approve(content.replace(fixtures.canaries[0], '[EMAIL_1]'));
    },
  });
  const saver = new MemorySaver();
  const create = (): {
    node: ToolNode;
    graph: Pick<Runnable<unknown, unknown>, 'invoke'>;
  } => {
    const node = new ToolNode({
      trace: true,
      tools: [lookup, approval],
      hookRegistry: hooks,
      humanInTheLoop: { enabled: true },
      maxToolResultChars: 128,
      toolOutputReferences: { enabled: true, maxOutputSize: 8192 },
      toolResultProtection: gate,
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('tools', node)
      .addEdge(START, 'tools')
      .addEdge('tools', END)
      .compile({ checkpointer: saver });
    return { node, graph };
  };
  const config = {
    configurable: { thread_id: 'mixed-ref-replay', run_id: 'mixed-ref-replay' },
  };
  const initial = create();
  const paused = await initial.graph.invoke(
    {
      messages: [
        new AIMessage({
          id: 'mixed-batch',
          content: '',
          tool_calls: [
            { id: 'lookup-control', name: 'lookup', args: { count: 42 } },
            {
              id: 'approval-control',
              name: 'approval_control',
              args: { count: 42 },
            },
          ],
        }),
      ],
    },
    config
  );
  expect(isInterrupted(paused)).toBe(true);
  expect(lookups).toBe(1);
  expect(approvals).toBe(0);
  if (!isInterrupted(paused)) throw new Error('Expected checkpointed approval');
  const replay = getToolBatchReplayState(paused.__interrupt__[0].value);
  expect(replay).toBeDefined();
  const resumedRuntime = create();
  const resumed = await resumedRuntime.graph.invoke(
    new Command({ resume: [{ type: 'approve' }] }),
    {
      configurable: { ...config.configurable, [TOOL_BATCH_REPLAY_KEY]: replay },
    }
  );
  expect(lookups).toBe(1);
  expect(approvals).toBe(1);
  expect(inspected).toEqual([raw, canonical]);
  expect(
    resumedRuntime.node
      ._unsafeGetToolOutputRegistry()!
      .get('mixed-ref-replay', 'tool0turn0')
  ).toBe(canonical);
  assertNoCanary(resumed);
});

it('protects text-only content_and_artifact exceptions through the native error lifecycle', async () => {
  const events: string[] = [];
  const inspected: string[] = [];
  const lookup = direct(() => {
    throw new Error(fixtures.canaries[0]);
  });
  lookup.responseFormat = 'content_and_artifact';
  const node = new ToolNode({
    trace: true,
    tools: [lookup],
    toolResultProtection: policy({
      inspect: ({ content, target }) => {
        inspected.push(content);
        expect(target.outcome).toBe('error');
        return approve('[EMAIL_1]');
      },
    }),
  });
  const result = (await node.invoke(state(), {
    callbacks: [observer(events)],
  })) as { messages: ToolMessage[] };
  expect(result.messages[0].status).toBe('error');
  expect(inspected).toEqual([fixtures.canaries[0]]);
  assertNoCanary(result);
  assertNoCanary(events.join(''));
  expect(events.join('')).toContain('[EMAIL_1]');
});

it.each([false, true])(
  'rejects hostile host envelopes before reading accessors (eager=%s)',
  async (eager) => {
    for (const key of [
      'toolCallId',
      'status',
      'content',
      'array-index',
      'own-content',
    ] as const) {
      let reads = 0;
      const completions: string[] = [];
      const getter = (): never => {
        reads++;
        throw new Error(fixtures.canaries[0]);
      };
      const result =
        (key === 'array-index' || key === 'own-content')
          ? {
            toolCallId: 'call-control',
            status: 'success',
            content: 'Allowed control',
          }
          : Object.create(
            Object.defineProperty({}, key, { get: getter }),
            Object.fromEntries(
              Object.entries({
                toolCallId: 'call-control',
                status: 'success',
                content: 'Allowed control',
              })
                .filter(([name]) => name !== key)
                .map(([name, value]) => [name, { value, enumerable: true }])
            )
          );
      if (key === 'own-content') Object.defineProperty(result, 'content', { get: getter, enumerable: true });
      const results = [result] as ToolExecuteResult[];
      if (key === 'array-index')
        Object.defineProperty(results, '0', { get: getter });
      const run = await Run.create({
        runId: `host-accessor-${eager}-${key}`,
        graphConfig: {
          type: 'standard',
          llmConfig: { provider: Providers.OPENAI },
          instructions: 'Allowed control.',
          toolDefinitions: [
            {
              name: 'lookup',
              parameters: {
                type: 'object',
                properties: { count: { type: 'number' } },
              },
            },
          ],
        },
        customHandlers: {
          [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
          [GraphEvents.ON_TOOL_EXECUTE]: {
            handle: (_event, data): void => {
              (data as ToolExecuteBatchRequest).resolve(results);
            },
          },
          [GraphEvents.ON_RUN_STEP_COMPLETED]: {
            handle: (_event, data): void => {
              completions.push(JSON.stringify(data));
            },
          },
        },
        eagerEventToolExecution: { enabled: eager },
        toolResultProtection: policy(),
        skipCleanup: true,
      });
      run.Graph!.overrideModel = new FakeChatModel({
        responses: ['', 'Must not continue'],
        toolCalls: [
          {
            id: 'call-control',
            name: 'lookup',
            args: { count: 42 },
            type: 'tool_call',
          },
        ],
      });
      const error = await run
        .processStream(
          { messages: [new HumanMessage('Allowed control')] },
          {
            version: 'v2',
            configurable: { thread_id: `host-accessor-${eager}-${key}` },
          }
        )
        .catch((value: Error) => value);
      expect(error).toMatchObject({ code: 'unsupported' });
      expect(reads).toBe(0);
      expect(completions).toEqual([]);
      assertNoCanary(String(error));
      assertNoCanary(run.Graph!.getRunMessages());
    }
  }
);

it.each(['python', 'bash'] as const)(
  'gates selected Cloudflare native outputs before any %s sandbox work',
  async (runtime) => {
    let executions = 0;
    const sandbox = {
      exec: async () => {
        executions++;
        return { exitCode: 0, stdout: 'Allowed control', stderr: '' };
      },
      readFile: async () => {
        executions++;
        return fixtures.canaries[0];
      },
      writeFile: async () => {
        executions++;
        return undefined;
      },
      mkdir: async () => undefined,
      listFiles: async () => [],
      deleteFile: async () => undefined,
    };
    const runner =
      runtime === 'python'
        ? createCloudflareProgrammaticToolCallingTool({ sandbox })
        : createCloudflareBashProgrammaticToolCallingTool({ sandbox });
    for (const manifest of [['read_file'], []]) {
      await expect(
        runner.invoke(
          {
            code:
              runtime === 'python'
                ? 'print(await read_file("canary.txt"))'
                : 'read_file \'{"path":"canary.txt"}\'',
            tool_manifest: manifest,
          },
          {
            toolCall: {
              id: 'cloudflare-control',
              name: runner.name,
              type: 'tool_call',
              args: {},
              toolDefs: [
                { name: 'read_file', allowed_callers: ['code_execution'] },
              ],
              toolResultProtection: policy({ toolNames: ['read_file'] }),
            },
          }
        )
      ).rejects.toMatchObject({ code: 'unsupported' });
    }
    expect(executions).toBe(0);
    const allowed = await runner.invoke(
      {
        code:
          runtime === 'python'
            ? 'print("Allowed control")'
            : 'echo "Allowed control"',
        tool_manifest: [],
      },
      {
        toolCall: {
          id: 'cloudflare-allowed',
          name: runner.name,
          type: 'tool_call',
          args: {},
          toolDefs: [],
          toolResultProtection: policy(),
        },
      }
    );
    expect(executions).toBeGreaterThan(0);
    expect(JSON.stringify(allowed)).toContain('Allowed control');
    assertNoCanary(allowed);
  }
);

it('keeps a late blocked local bridge result terminal while draining detached requests', async () => {
  const started = deferred<void>();
  const decision = deferred<ToolResultProtectionResult>();
  const runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
  const pending = runner.invoke(
    {
      lang: 'bash',
      code: 'lookup \'{"count":42}\' >/dev/null 2>&1 &\nsleep 0.05\necho "Allowed outer control"',
      tool_manifest: ['lookup'],
    },
    {
      toolCall: {
        id: 'late-local-control',
        name: runner.name,
        type: 'tool_call',
        args: {},
        toolMap: new Map([['lookup', direct(() => fixtures.canaries[0])]]),
        toolDefs: [
          {
            name: 'lookup',
            allowed_callers: ['code_execution'],
            parameters: {
              type: 'object',
              properties: { count: { type: 'number' } },
            },
          },
        ],
        toolResultProtection: policy({
          inspect: () => {
            started.resolve();
            return decision.promise;
          },
        }),
      },
    }
  );
  const outcome = pending.catch((error: Error) => error);
  await started.promise;
  await new Promise((resolve) => setTimeout(resolve, 100));
  decision.resolve({ version: 1, ok: false, error: { code: 'blocked' } });
  expect(await outcome).toMatchObject({ code: 'blocked' });
});

it.each([false, true])(
  'inspects full direct approval response before model truncation (block=%s)',
  async (block) => {
    const raw = 'Allowed response '.repeat(80) + fixtures.canaries[0];
    const inspected: string[] = [];
    let executions = 0;
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async () => ({ decision: 'ask', reason: 'Allowed approval control' }),
      ],
    });
    const node = new ToolNode({
      tools: [
        direct(() => {
          executions++;
          return 'Must not run';
        }),
      ],
      hookRegistry: hooks,
      humanInTheLoop: { enabled: true },
      maxToolResultChars: 128,
      toolResultProtection: policy({
        inspect: ({ content }) => {
          inspected.push(content);
          return block
            ? { version: 1, ok: false, error: { code: 'blocked' } }
            : approve('[EMAIL_1]');
        },
      }),
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('tools', node)
      .addEdge(START, 'tools')
      .addEdge('tools', END)
      .compile({ checkpointer: new MemorySaver() });
    const config = {
      configurable: {
        thread_id: `response-c1-${block}`,
        run_id: `response-c1-${block}`,
      },
    };
    expect(isInterrupted(await graph.invoke(state(), config))).toBe(true);
    const resumed = graph.invoke(
      new Command({ resume: [{ type: 'respond', responseText: raw }] }),
      config
    );
    if (block) await expect(resumed).rejects.toMatchObject({ code: 'blocked' });
    else {
      const value = await resumed;
      assertNoCanary(value);
      expect(JSON.stringify(value)).toContain('[EMAIL_1]');
    }
    expect(executions).toBe(0);
    expect(inspected).toEqual([raw]);
  }
);

it('re-inspects an observational error-handler replacement before completion and model reuse', async () => {
  const inspected: string[] = []; const events: string[] = [];
  const node = new ToolNode({ trace: true, tools: [direct(() => { throw new Error('Allowed initial error'); })], toolCallStepIds: new Map([['call-control', 'step-control']]), toolResultProtection: policy({ inspect: ({ content }) => { inspected.push(content); return approve(content.replace(fixtures.canaries[0], '[EMAIL_1]')); } }), errorHandler: async ({ error }) => { error!.message = fixtures.canaries[0]; return false; } });
  const result = await node.invoke(state(), { callbacks: [observer(events)] });
  expect(inspected).toEqual(['Allowed initial error', fixtures.canaries[0]]); assertNoCanary(result); assertNoCanary(events.join('')); expect(events.join('')).toContain('[EMAIL_1]');
});

it.each(['lookup', 'request_alias'])('rejects uncertified nested executable aliases before execution (selected=%s)', async (selected) => {
  let executions = 0;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const result = executeTools([{ id: 'alias-control', name: 'request_alias', input: { count: 42 } }], new Map([['request_alias', lookup]]), 'run_tools_with_code', { policy: policy({ toolNames: [selected] }) });
  await expect(result).rejects.toMatchObject({ code: 'unsupported' });
  expect(executions).toBe(0);
});

it('rejects a standalone nested incompatible policy before native tool execution', async () => {
  let executions = 0;
  const future = { ...policy(), version: 2 } as unknown as ToolResultProtection;
  await expect(executeTools([{ id: 'version-control', name: 'lookup', input: { count: 42 } }], new Map([['lookup', direct(() => { executions++; return fixtures.canaries[0]; })]]), 'run_tools_with_code', { policy: future })).rejects.toMatchObject({ code: 'incompatible' });
  expect(executions).toBe(0);
});

it('keeps disconnected selected bridge work terminal after the enclosing local process exits', async () => {
  const started = deferred<void>(); const decision = deferred<ToolResultProtectionResult>();
  const runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
  const pending = runner.invoke({ lang: 'bash', code: 'curl -sS --max-time 0.05 -X POST -H "Content-Type: application/json" -H "$__LIBRECHAT_TOOL_HEADER: $__LIBRECHAT_TOOL_TOKEN" --data-binary \'{"name":"lookup","input":{"count":42}}\' "$__LIBRECHAT_TOOL_BRIDGE?mode=text" >/dev/null 2>&1 || true\necho "Allowed outer control"', tool_manifest: ['lookup'] }, { toolCall: {
    id: 'disconnect-local-control', name: runner.name, type: 'tool_call', args: {}, toolMap: new Map([['lookup', direct(() => fixtures.canaries[0])]]), toolDefs: [{ name: 'lookup', allowed_callers: ['code_execution'], parameters: { type: 'object', properties: { count: { type: 'number' } } } }],
    toolResultProtection: policy({ inspect: () => { started.resolve(); return decision.promise; } }),
  } });
  const outcome = pending.catch((error: Error) => error);
  await started.promise; await new Promise((resolve) => setTimeout(resolve, 100));
  decision.resolve({ version: 1, ok: false, error: { code: 'blocked' } });
  expect(await outcome).toMatchObject({ code: 'blocked' });
});

it.each((['legacy', 'protected', 'unselected', 'new-selection', 'reblocked'] as const).flatMap((mode) => (['direct', 'host'] as const).map((path) => [mode, path] as const)))('gates earlier-turn checkpoint references before resumed reuse (%s/%s)', async (mode, path) => {
  const runId = `earlier-ref-${mode}-${path}`; const config = { configurable: { run_id: runId, thread_id: runId } };
  const registry = new ToolOutputReferenceRegistry(); const inspected: string[] = []; const piped: string[] = [];
  const sourceName = mode === 'unselected' ? 'other' : 'lookup';
  const source = new DynamicStructuredTool({ name: sourceName, description: 'Allowed source control.', schema: z.object({ count: z.number() }), func: async () => mode === 'unselected' ? 'Allowed unselected control' : fixtures.canaries[0] });
  const gate = policy({ inspect: ({ content }) => { inspected.push(content); return approve(content.replace(fixtures.canaries[0], '[EMAIL_1]')); } });
  let previous: ToolResultProtection | undefined = gate;
  if (mode === 'legacy') previous = undefined;
  if (mode === 'new-selection') previous = policy({ toolNames: ['other'] });
  const producer = new ToolNode({ trace: true, tools: [source], eventDrivenMode: path === 'host', toolOutputRegistry: registry, toolResultProtection: previous });
  await producer.invoke({ messages: [new AIMessage({ id: 'previous-turn', content: '', tool_calls: [{ id: 'source-control', name: sourceName, args: { count: 42 } }] })] }, { ...config, callbacks: [observer([], (request) => request.resolve([{ toolCallId: 'source-control', status: 'success', content: mode === 'unselected' ? 'Allowed unselected control' : fixtures.canaries[0] }]))] });
  const pipe = new DynamicStructuredTool({ name: 'pipe', description: 'Allowed pipe.', schema: z.object({ command: z.string() }), func: async ({ command }) => { piped.push(command); return command; } });
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval control' })] });
  const saver = new MemorySaver();
  const create = (resultPolicy: ToolResultProtection | undefined, refs: ToolOutputReferenceRegistry): Pick<Runnable<unknown, unknown>, 'invoke'> => new StateGraph(MessagesAnnotation).addNode('tools', new ToolNode({ tools: [pipe], toolOutputRegistry: refs, toolResultProtection: resultPolicy, hookRegistry: hooks, humanInTheLoop: { enabled: true } })).addEdge(START, 'tools').addEdge('tools', END).compile({ checkpointer: saver });
  const initial = create(previous, registry);
  const paused = await initial.invoke({ messages: [new AIMessage({ id: 'next-turn', content: '', tool_calls: [{ id: 'pipe-control', name: 'pipe', args: { command: '{{' + 'tool0turn0' + '}}' } }] })] }, config);
  if (!isInterrupted(paused)) throw new Error('Expected approval checkpoint');
  const replay = getToolBatchReplayState(paused.__interrupt__[0].value); expect(replay?.records[0].referenceState?.entries).toHaveLength(1);
  const resumePolicy = mode === 'reblocked' ? policy({ inspect: () => ({ version: 1, ok: false, error: { code: 'blocked' } }) }) : gate;
  const resumed = create(resumePolicy, new ToolOutputReferenceRegistry()).invoke(new Command({ resume: [{ type: 'approve' }] }), { configurable: { ...config.configurable, [TOOL_BATCH_REPLAY_KEY]: replay } });
  if (mode === 'legacy' || mode === 'new-selection' || mode === 'reblocked') {
    await expect(resumed).rejects.toMatchObject({ code: mode === 'reblocked' ? 'blocked' : 'incompatible' }); expect(piped).toEqual([]); return;
  }
  const value = await resumed; assertNoCanary(value); expect(piped).toEqual([mode === 'unselected' ? 'Allowed unselected control' : '[EMAIL_1]']);
  expect(inspected).toEqual(mode === 'unselected' ? [] : [fixtures.canaries[0], '[EMAIL_1]']);
});

it.each([false, true])('preserves unselected host outcome fields in protected routing (mixed=%s)', async (mixed) => {
  const events: string[] = [];
  const other = new DynamicStructuredTool({ name: 'other', description: 'Allowed control.', schema: z.object({ count: z.number() }), func: async () => 'Unused schema control' });
  const node = new ToolNode({ trace: true, tools: [direct(() => 'Unused schema control'), other], eventDrivenMode: true, toolResultProtection: policy(), toolCallStepIds: new Map([['other-control', 'other-step'], ['call-control', 'lookup-step']]) });
  const calls = [{ id: 'other-control', name: 'other', args: { count: 42, intent: 'Searching' } }, ...(mixed ? [{ id: 'call-control', name: 'lookup', args: { count: 42 } }] : [])];
  const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: calls })] }, { callbacks: [observer(events, (request) => request.resolve(request.toolCalls.map((call) => call.name === 'other' ? { toolCallId: call.id, status: 'success', content: 'Allowed unselected output', outcome: 'Searched', outcome_patch: { from: 'Searching', to: 'Searched' } } : { toolCallId: call.id, status: 'success', content: fixtures.canaries[0] })))] });
  expect(events.join('')).toContain('Searched'); expect(JSON.stringify(result)).toContain('Allowed unselected output'); assertNoCanary(result); assertNoCanary(events.join(''));
});

it.each([false, true])('preserves unselected host outcomes through real run/eager dispatch (eager=%s)', async (eager) => {
  const completions: string[] = [];
  const run = await Run.create({ runId: `outcomes-${eager}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, instructions: 'Allowed control.', toolDefinitions: ['lookup', 'other'].map((name) => ({ name, parameters: { type: 'object', properties: { count: { type: 'number' }, intent: { type: 'string' } } } })) }, toolResultProtection: policy(), eagerEventToolExecution: { enabled: eager }, customHandlers: {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_TOOL_EXECUTE]: { handle: (_event, data): void => { const request = data as ToolExecuteBatchRequest; request.resolve(request.toolCalls.map((call) => call.name === 'other' ? { toolCallId: call.id, content: 'Allowed unselected control', status: 'success', outcome_patch: { from: 'Searching', to: 'Searched' } } : { toolCallId: call.id, content: fixtures.canaries[0], status: 'success' })); } },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } },
  }, skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed final control'], toolCalls: [{ id: 'other-control', name: 'other', args: { count: 42, intent: 'Searching' }, type: 'tool_call' }, { id: 'lookup-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `outcomes-${eager}` } });
  expect(completions.join('')).toContain('Searched'); assertNoCanary(completions.join('')); assertNoCanary(run.Graph!.getRunMessages());
});

it('rejects selected host outcome aliases rather than admitting unchecked public text', async () => {
  const events: string[] = [];
  const node = new ToolNode({ trace: true, tools: [direct(() => 'Unused schema control')], eventDrivenMode: true, toolResultProtection: policy() });
  await expect(node.invoke(state(), { callbacks: [observer(events, (request) => request.resolve([{ toolCallId: 'call-control', content: 'Allowed control', status: 'success', outcome: fixtures.canaries[0] }]))] })).rejects.toMatchObject({ code: 'unsupported' });
  assertNoCanary(events.join(''));
});

it.each([false, true])('rejects host identity mutation before any protected publication (eager=%s)', async (eager) => {
  for (const field of ['name', 'id', 'array'] as const) {
    for (const timing of ['before', 'inspecting', 'cleanup'] as const) {
      const inspected = deferred<void>(); const decision = deferred<ToolResultProtectionResult>(); const completions: string[] = []; let executions = 0;
      const mutate = (request: ToolExecuteBatchRequest): void => {
        if (field === 'name') request.toolCalls[0].name = 'host_alias';
        if (field === 'id') request.toolCalls[0].id = 'host-id-alias';
        if (field === 'array') request.toolCalls = [{ ...request.toolCalls[0], name: 'host_alias' }];
      };
      const run = await Run.create({ runId: `host-identity-${eager}-${field}-${timing}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, instructions: 'Allowed control.', toolDefinitions: [{ name: 'lookup', parameters: { type: 'object', properties: { count: { type: 'number' } } } }] }, toolResultProtection: policy({ inspect: () => { inspected.resolve(); return timing === 'before' ? approve('[EMAIL_1]') : decision.promise; } }), eagerEventToolExecution: { enabled: eager }, customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_TOOL_EXECUTE]: { handle: async (_event, data): Promise<void> => {
          const request = data as ToolExecuteBatchRequest; executions++;
          if (timing === 'before') mutate(request);
          request.resolve([{ toolCallId: 'call-control', status: 'success', content: fixtures.canaries[0] }]);
          if (timing !== 'before') { await inspected.promise; if (timing === 'cleanup') { decision.resolve(approve('[EMAIL_1]')); await new Promise((resolve) => setImmediate(resolve)); } mutate(request); decision.resolve(approve('[EMAIL_1]')); }
        } },
        [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } },
      }, skipCleanup: true });
      run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: [{ id: 'call-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
      await expect(run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `host-identity-${eager}-${field}-${timing}` } })).rejects.toMatchObject({ code: 'unsupported' });
      expect(executions).toBe(1); expect(completions).toEqual([]); assertNoCanary(run.Graph!.getRunMessages());
    }
  }
});

it.each(['blocked', 'timeout'] as const)('prioritizes a required %s failure over an interrupting sibling approval', async (code) => {
  let executions = 0; let regularExecutions = 0; const inspected: string[] = [];
  const ask = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed pause control.', schema: z.object({ count: z.number() }), func: async () => { throw new GraphInterrupt([{ id: 'approval-control', value: 'Allowed approval control' }]); } });
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const regular = new DynamicStructuredTool({ name: 'regular_control', description: 'Must not run after a safety failure.', schema: z.object({ count: z.number() }), func: async () => { regularExecutions++; return 'Must not run'; } });
  const gate = policy({ timeoutMs: code === 'timeout' ? 30 : 10000, inspect: ({ content }) => { inspected.push(content); return code === 'blocked' ? { version: 1, ok: false, error: { code: 'blocked' } } : new Promise<ToolResultProtectionResult>(() => {}); } });
  const node = new ToolNode({ tools: [ask, lookup, regular], interruptingToolNames: new Set(['approval_control', 'lookup']), toolResultProtection: gate });
  const graph = new StateGraph(MessagesAnnotation).addNode('tools', node).addEdge(START, 'tools').addEdge('tools', END).compile({ checkpointer: new MemorySaver() });
  await expect(graph.invoke({ messages: [new AIMessage({ id: 'interrupting-batch', content: '', tool_calls: [{ id: 'approval-control', name: 'approval_control', args: { count: 42 } }, { id: 'call-control', name: 'lookup', args: { count: 42 } }, { id: 'regular-control', name: 'regular_control', args: { count: 42 } }] })] }, { configurable: { thread_id: `interrupting-protection-${code}`, run_id: `interrupting-protection-${code}` } })).rejects.toMatchObject({ code });
  expect(executions).toBe(1); expect(regularExecutions).toBe(0); expect(inspected).toEqual([fixtures.canaries[0]]);
});

it.each(([false, true] as const).flatMap((eager) => (['throw', 'absent'] as const).map((mode) => [eager, mode] as const)))('fails raw-free for a required host handler %s/%s', async (eager, mode) => {
  const completions: string[] = [];
  const run = await Run.create({ runId: `throwing-host-${eager}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, instructions: 'Allowed control.', toolDefinitions: [{ name: 'lookup', parameters: { type: 'object', properties: { count: { type: 'number' } } } }] }, toolResultProtection: policy(), eagerEventToolExecution: { enabled: eager }, customHandlers: {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    ...(mode === 'throw' ? { [GraphEvents.ON_TOOL_EXECUTE]: { handle: (): never => { throw new Error(fixtures.canaries[0]); } } } : {}),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } },
  }, skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: [{ id: 'call-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  const error = await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `throwing-host-${eager}` } }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unavailable' }); assertNoCanary(String(error)); expect(completions).toEqual([]); assertNoCanary(run.Graph!.getRunMessages());
});

it.each(['name', 'format'] as const)('rejects native producer %s identity changes before callback publication', async (field) => {
  const events: string[] = []; let executions = 0;
  const lookup = direct(() => { executions++; if (field === 'name') Object.defineProperty(lookup, 'name', { value: 'producer_alias' }); else lookup.responseFormat = 'content_and_artifact'; return fixtures.canaries[0]; });
  const node = new ToolNode({ trace: true, tools: [lookup], toolResultProtection: policy() });
  await expect(node.invoke(state(), { callbacks: [observer(events)] })).rejects.toMatchObject({ code: 'unsupported' });
  expect(executions).toBe(1); assertNoCanary(events.join(''));
});

it.each([false, true])('keeps required child-host throws raw-free and terminal (eager=%s)', async (eager) => {
  let executions = 0; const updates: string[] = [];
  const run = await Run.create({ runId: `child-host-throw-${eager}`, graphConfig: { type: 'standard', agents: [{ agentId: 'parent', provider: Providers.OPENAI, instructions: 'Allowed parent control.', subagentConfigs: [{ type: 'worker', name: 'Worker', description: 'Allowed child.', agentInputs: { agentId: 'child', provider: Providers.OPENAI, instructions: 'Allowed child control.', toolDefinitions: [{ name: 'lookup', parameters: { type: 'object', properties: { count: { type: 'number' } } } }] } }] }] }, toolResultProtection: policy(), eagerEventToolExecution: { enabled: eager }, customHandlers: {
    [GraphEvents.ON_TOOL_EXECUTE]: { handle: (): never => { executions++; throw new Error(fixtures.canaries[0]); } },
    [GraphEvents.ON_SUBAGENT_UPDATE]: { handle: (_event, data): void => { updates.push(JSON.stringify(data)); } },
  }, skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: [{ id: 'child-control', name: Constants.SUBAGENT, args: { description: 'Allowed task.', subagent_type: 'worker' }, type: 'tool_call' }] });
  run.Graph!.setSubagentModelOverride(new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: [{ id: 'lookup-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] }));
  const error = await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `child-host-throw-${eager}` } }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unavailable' }); expect(executions).toBe(1); assertNoCanary(String(error)); assertNoCanary(updates.join('')); assertNoCanary(run.Graph!.getRunMessages());
});

it.each(['live', 'restored-same', 'restored-strict', 'live-strict', 'blocked'] as const)('binds reference reuse to the current policy rather than a serialized protected flag (%s)', async (mode) => {
  const runId = `restored-reference-${mode}`; const config = { configurable: { run_id: runId } }; const inspected: string[] = []; const piped: string[] = [];
  const previous = policy({ inspect: ({ content }) => { inspected.push(content); return approve(content); } });
  const registry = new ToolOutputReferenceRegistry();
  const source = new ToolNode({ tools: [direct(() => fixtures.canaries[0])], toolOutputRegistry: registry, toolResultProtection: previous });
  await source.invoke(state(), config);
  const restored = mode.startsWith('restored') || mode === 'blocked';
  const consumerRegistry = restored ? new ToolOutputReferenceRegistry() : registry;
  if (restored) consumerRegistry.restoreState(runId, JSON.parse(JSON.stringify(registry.snapshotState(runId))));
  const gate = mode === 'live' || mode === 'restored-same' ? previous : policy({ inspect: ({ content }) => { inspected.push(content); return mode === 'blocked' ? { version: 1, ok: false, error: { code: 'blocked' } } : approve('[EMAIL_1]'); } });
  const pipe = new DynamicStructuredTool({ name: 'pipe', description: 'Allowed pipe control.', schema: z.object({ command: z.string() }), func: async ({ command }) => { piped.push(command); return command; } });
  const node = new ToolNode({ tools: [pipe], toolOutputRegistry: consumerRegistry, toolResultProtection: gate });
  const pending = node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'pipe-control', name: 'pipe', args: { command: '{{' + 'tool0turn0' + '}}' } }] })] }, config);
  if (mode === 'blocked') { await expect(pending).rejects.toMatchObject({ code: 'blocked' }); expect(piped).toEqual([]); return; }
  const result = await pending;
  expect(inspected).toEqual(mode === 'live' ? [fixtures.canaries[0]] : [fixtures.canaries[0], fixtures.canaries[0]]);
  expect(piped).toEqual([mode === 'live' || mode === 'restored-same' ? fixtures.canaries[0] : '[EMAIL_1]']);
  if (mode !== 'live' && mode !== 'restored-same') { assertNoCanary(result); expect(consumerRegistry.get(runId, 'tool0turn0')).toBe('[EMAIL_1]'); }
  const before = inspected.length;
  await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'pipe-again', name: 'pipe', args: { command: '{{' + 'tool0turn0' + '}}' } }] })] }, config);
  expect(inspected).toHaveLength(before);
  expect(piped[1]).toBe(piped[0]);
});

it.each(['id', 'name', 'tool_call_id', 'status', 'content', 'metadata', 'metadata-object', 'artifact', 'response-metadata', 'accessor'] as const)('rejects a retained producer message %s mutation during inspection before native release', async (field) => {
  const started = deferred<void>(); const decision = deferred<ToolResultProtectionResult>(); const events: string[] = [];
  const message = new ToolMessage({ id: 'call-control', name: 'lookup', tool_call_id: 'call-control', status: 'success', content: fixtures.canaries[0] });
  const node = new ToolNode({ trace: true, tools: [direct(() => message)], toolResultProtection: policy({ inspect: () => { started.resolve(); return decision.promise; } }), toolOutputReferences: { enabled: true } });
  const outcome = node.invoke(state(), { configurable: { run_id: `message-mutation-${field}` }, callbacks: [observer(events)] }).catch((error: Error) => error);
  await started.promise;
  if (field === 'id') message.id = fixtures.canaries[0];
  if (field === 'name') message.name = fixtures.canaries[0];
  if (field === 'tool_call_id') message.tool_call_id = fixtures.canaries[0];
  if (field === 'status') message.status = 'error';
  if (field === 'content') message.content = 'Different retained producer control';
  if (field === 'metadata') message.additional_kwargs._refScope = fixtures.canaries[0];
  if (field === 'metadata-object') message.additional_kwargs = { _refScope: fixtures.canaries[0] };
  if (field === 'artifact') message.artifact = { value: fixtures.canaries[0] };
  if (field === 'response-metadata') message.response_metadata = { output: fixtures.canaries[0] };
  if (field === 'accessor') Object.defineProperty(message, 'id', { get: () => { throw new Error(fixtures.canaries[0]); } });
  decision.resolve(approve('[EMAIL_1]'));
  const error = await outcome;
  expect(error).toMatchObject({ code: 'unsupported' }); assertNoCanary(String(error)); assertNoCanary(events.join('')); expect(events.join('')).not.toContain('[EMAIL_1]');
  expect(node._unsafeGetToolOutputRegistry()!.size).toBe(0);
});

it.each(['success', 'error'] as const)('releases an immutable producer message with its original identity/status (%s)', async (status) => {
  const events: string[] = []; const message = new ToolMessage({ id: 'call-control', name: 'lookup', tool_call_id: 'call-control', status, content: fixtures.canaries[0] });
  const node = new ToolNode({ trace: true, tools: [direct(() => message)], toolResultProtection: policy() });
  const result = await node.invoke(state(), { callbacks: [observer(events)] }) as { messages: ToolMessage[] };
  expect(result.messages[0]).toMatchObject({ id: 'call-control', tool_call_id: 'call-control', name: 'lookup', status });
  expect(result.messages[0].content).toBe('[EMAIL_1]'); assertNoCanary(result); assertNoCanary(events.join('')); expect(events.join('')).toContain('[EMAIL_1]');
});

it.each(['id', 'status', 'metadata'] as const)('rechecks canonical message %s provenance before storage and reuse', async (field) => {
  const events: string[] = [];
  const callback = observer(events);
  const observe = callback.handleToolEnd!.bind(callback);
  callback.handleToolEnd = async (output, ...args): Promise<void> => {
    await observe(output, ...args);
    if (!(output instanceof ToolMessage)) throw new Error('Expected native ToolMessage');
    if (field === 'id') output.id = fixtures.canaries[0];
    if (field === 'status') output.status = 'error';
    if (field === 'metadata') output.additional_kwargs._refScope = fixtures.canaries[0];
  };
  const node = new ToolNode({ trace: true, tools: [direct(() => fixtures.canaries[0])], toolResultProtection: policy(), toolOutputReferences: { enabled: true } });
  await expect(node.invoke(state(), { configurable: { run_id: `canonical-mutation-${field}` }, callbacks: [callback] })).rejects.toMatchObject({ code: 'unsupported' });
  assertNoCanary(events.join('')); expect(node._unsafeGetToolOutputRegistry()!.size).toBe(0);
});

it('copies retained producer metadata before canonical native publication', async () => {
  const original = new ToolMessage({ id: 'call-control', tool_call_id: 'call-control', content: fixtures.canaries[0] });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => {
    if (!(output instanceof ToolMessage)) throw new Error('Expected native ToolMessage');
    expect(output.additional_kwargs).not.toBe(original.additional_kwargs);
    original.id = fixtures.canaries[0]; original.status = 'error'; original.additional_kwargs._refScope = fixtures.canaries[0];
  } }); callback.awaitHandlers = true;
  const node = new ToolNode({ trace: true, tools: [direct(() => original)], toolResultProtection: policy() });
  const result = await node.invoke(state(), { callbacks: [callback] }) as { messages: ToolMessage[] };
  expect(result.messages[0]).toMatchObject({ id: 'call-control', status: 'success', content: '[EMAIL_1]' }); assertNoCanary(result);
});

it.each((['text', 'json'] as const).flatMap((mode) => (['ordinary', 'string', 'typed', 'prepared', 'stream'] as const).map((reasonKind) => [mode, reasonKind] as const)))('normalizes and latches local bridge cancellation before the %s/%s response', async (mode, reasonKind) => {
  const responses: string[] = []; let executions = 0; let inspections = 0; const received = deferred<void>();
  const audit = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk: Buffer): void => { body += chunk.toString(); });
    request.on('end', (): void => { responses.push(body); response.end('Allowed audit control'); received.resolve(); });
  });
  await new Promise<void>((resolve) => audit.listen(0, '127.0.0.1', resolve));
  let reason: Error | string = new Error(fixtures.canaries[0]);
  if (reasonKind === 'string') reason = fixtures.canaries[0];
  if (reasonKind === 'typed') { reason = new ToolResultProtectionError('blocked'); reason.message = fixtures.canaries[0]; }
  if (reasonKind === 'prepared') reason = new PreparedSubagentError('Allowed prepared control');
  if (reasonKind === 'stream') reason = new StreamLimitExceededError({ kind: 'tool_call_args', limit: 10, observed: 11, toolName: 'lookup' });
  const controller = new AbortController(); controller.abort(reason);
  const runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
  const invoke: (params: { lang: string; code: string; tool_manifest: string[] }, manager: undefined, config: RunnableConfig & { toolCall?: ToolCall & Partial<ProgrammaticCache> }) => Promise<unknown> = Reflect.get(runner, '_call');
  try {
    const auditUrl = `http://127.0.0.1:${(audit.address() as AddressInfo).port}`;
    const code = `response=$(curl -sS -X POST -H "Content-Type: application/json" -H "$__LIBRECHAT_TOOL_HEADER: $__LIBRECHAT_TOOL_TOKEN" --data-binary '{"name":"lookup","input":{"count":42}}' "$__LIBRECHAT_TOOL_BRIDGE?mode=${mode}")
printf '%s' "$response" | curl -sS -X POST --data-binary @- '${auditUrl}'
echo "Allowed outer control"`;
    // Exercise the real runner body so LangChain's outer abort race cannot mask bridge responses.
    const outcome = invoke.call(runner, { lang: 'bash', code, tool_manifest: ['lookup'] }, undefined, { signal: controller.signal, toolCall: { id: 'cancelled-local', name: runner.name, type: 'tool_call', args: {}, toolMap: new Map([['lookup', direct(() => { executions++; return fixtures.canaries[0]; })]]), toolDefs: [{ name: 'lookup', allowed_callers: ['code_execution'], parameters: { type: 'object', properties: { count: { type: 'number' } } } }], toolResultProtection: policy({ inspect: () => { inspections++; return approve('[EMAIL_1]'); } }) } }).catch((value: Error) => value);
    await received.promise; expect(responses).toHaveLength(1); assertNoCanary(responses.join('')); const error = await outcome;
    if (reasonKind === 'prepared' || reasonKind === 'stream') expect(error).toBe(reason);
    else { const code = reasonKind === 'typed' ? 'blocked' : 'cancelled'; expect(error).toMatchObject({ code }); expect(responses[0]).toContain(code); } assertNoCanary(String(error)); expect(executions).toBe(0); expect(inspections).toBe(0);
  } finally { audit.closeAllConnections(); await new Promise<void>((resolve) => audit.close(() => resolve())); }
});

it.each(['map', 'some', 'iterator', 'subclass', 'proxy', 'iterable', 'index', 'sparse', 'symbol'] as const)('rejects checkpoint-controlled reference collection %s before execution', async (kind) => {
  let calls = 0; let inspections = 0; let executions = 0;
  const entries: ToolOutputReferenceState['entries'] = [{ key: 'tool0turn0', value: fixtures.canaries[0], protection: { version: 1, toolName: 'lookup', toolCallId: 'source-control', protected: true } }];
  let hostile = entries;
  if (kind === 'map') Object.defineProperty(hostile, 'map', { value: () => { calls++; return entries; } });
  if (kind === 'some') Object.defineProperty(hostile, 'some', { value: () => { calls++; return false; } });
  if (kind === 'iterator') Object.defineProperty(hostile, Symbol.iterator, { value: function* () { calls++; yield entries[0]; } });
  if (kind === 'subclass') { class ReferenceArray extends Array<ToolOutputReferenceState['entries'][number]> {} hostile = new ReferenceArray(entries[0]); }
  if (kind === 'proxy') hostile = new Proxy(entries, { get: (target, key, receiver): unknown => { calls++; return Reflect.get(target, key, receiver); } });
  if (kind === 'iterable') hostile = { [Symbol.iterator]: function* () { calls++; yield entries[0]; }, some: () => { calls++; return false; } } as unknown as typeof entries;
  if (kind === 'index') Object.defineProperty(hostile, '0', { get: () => { calls++; return entries[0]; } });
  if (kind === 'sparse') hostile = new Array<ToolOutputReferenceState['entries'][number]>(1);
  if (kind === 'symbol') Object.defineProperty(hostile, Symbol('raw-alias'), { value: fixtures.canaries[0], enumerable: true });
  const snapshot: ToolOutputReferenceState = { entries: hostile, turnCounter: 1, warnedNonStringTools: [] };
  class CheckpointRegistry extends ToolOutputReferenceRegistry {
    snapshotState(): ToolOutputReferenceState { return snapshot; }
  }
  const registry = new CheckpointRegistry();
  const pipe = new DynamicStructuredTool({ name: 'pipe', description: 'Allowed consumer.', schema: z.object({ command: z.string() }), func: async ({ command }) => { executions++; return command; } });
  const events: string[] = [];
  const node = new ToolNode({ trace: true, tools: [pipe], toolOutputRegistry: registry, toolResultProtection: policy({ inspect: ({ content }) => { inspections++; return approve(content.replace(fixtures.canaries[0], '[EMAIL_1]')); } }) });
  const error = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'pipe-control', name: 'pipe', args: { command: '{{' + 'tool0turn0' + '}}' } }] })] }, { configurable: { run_id: `hostile-reference-${kind}` }, callbacks: [observer(events)] }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unsupported' }); expect(calls).toBe(0); expect(executions).toBe(0); expect(inspections).toBe(0); assertNoCanary(String(error)); assertNoCanary(events.join('')); expect(registry.size).toBe(0);
});

it.each((['additional', 'response'] as const).flatMap((field) => [true, false].map((enumerable) => [field, enumerable] as const)))('rejects %s symbol metadata before native callbacks (enumerable=%s)', async (field, enumerable) => {
  const alias = Symbol('producer-alias'); const events: string[] = []; let inspections = 0; let released = 0;
  const message = new ToolMessage({ tool_call_id: 'call-control', content: 'Allowed content control' });
  Object.defineProperty(field === 'additional' ? message.additional_kwargs : message.response_metadata, alias, { value: fixtures.canaries[0], enumerable });
  const callback = observer(events);
  callback.handleToolEnd = (output): void => {
    released++;
    if (!(output instanceof ToolMessage)) return;
    for (const metadata of [output.additional_kwargs, output.response_metadata]) {
      for (const key of Reflect.ownKeys(metadata)) events.push(String(Reflect.get(metadata, key)));
    }
  };
  const node = new ToolNode({ trace: true, tools: [direct(() => message)], toolResultProtection: policy({ inspect: () => { inspections++; return approve('Allowed canonical control'); } }), toolOutputReferences: { enabled: true } });
  await expect(node.invoke(state(), { callbacks: [callback] })).rejects.toMatchObject({ code: 'unsupported' });
  expect(inspections).toBe(0); expect(released).toBe(0); assertNoCanary(events.join('')); expect(node._unsafeGetToolOutputRegistry()!.size).toBe(0);
});

it('keeps admitted reference identity and metadata detached while current-policy inspection awaits', async () => {
  const started = deferred<void>(); const decision = deferred<ToolResultProtectionResult>(); const piped: string[] = []; const inspected: string[] = [];
  const snapshot: ToolOutputReferenceState = { entries: [{ key: 'tool0turn0', value: fixtures.canaries[0], protection: { version: 1, toolName: 'lookup', toolCallId: 'source-control', protected: true } }], turnCounter: 1, warnedNonStringTools: [] };
  class CheckpointRegistry extends ToolOutputReferenceRegistry { snapshotState(): ToolOutputReferenceState { return snapshot; } }
  const registry = new CheckpointRegistry();
  const pipe = new DynamicStructuredTool({ name: 'pipe', description: 'Allowed consumer.', schema: z.object({ command: z.string() }), func: async ({ command }) => { piped.push(command); return command; } });
  const node = new ToolNode({ tools: [pipe], toolOutputRegistry: registry, toolResultProtection: policy({ inspect: ({ content, toolName, toolCallId }) => { inspected.push(content); expect(toolName).toBe('lookup'); expect(toolCallId).toBe('source-control'); started.resolve(); return decision.promise; } }) });
  const outcome = node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'pipe-control', name: 'pipe', args: { command: '{{' + 'tool0turn0' + '}}' } }] })] }, { configurable: { run_id: 'detached-reference' } });
  await started.promise;
  snapshot.entries[0].value = 'Unchecked producer replacement'; snapshot.entries[0].key = 'tool0turn99'; snapshot.entries[0].protection!.toolName = 'unselected_alias'; snapshot.warnedNonStringTools.push(fixtures.canaries[0]);
  decision.resolve(approve('[EMAIL_1]'));
  const result = await outcome; expect(inspected).toEqual([fixtures.canaries[0]]); expect(piped).toEqual(['[EMAIL_1]']); expect(registry.get('detached-reference', 'tool0turn0')).toBe('[EMAIL_1]'); expect(registry.get('detached-reference', 'tool0turn99')).toBeUndefined(); assertNoCanary(result);
});

it.each(['state', 'entry', 'source'] as const)('rejects a hidden checkpoint %s symbol before reference reuse', async (field) => {
  let inspections = 0; let executions = 0;
  const snapshot: ToolOutputReferenceState = { entries: [{ key: 'tool0turn0', value: 'Allowed control', protection: { version: 1, toolName: 'lookup', toolCallId: 'source-control', protected: true } }], turnCounter: 1, warnedNonStringTools: [] };
  let metadata: object = snapshot;
  if (field === 'entry') metadata = snapshot.entries[0];
  if (field === 'source') metadata = snapshot.entries[0].protection!;
  Object.defineProperty(metadata, Symbol('raw-alias'), { value: fixtures.canaries[0] });
  class CheckpointRegistry extends ToolOutputReferenceRegistry { snapshotState(): ToolOutputReferenceState { return snapshot; } }
  const node = new ToolNode({ tools: [direct(() => { executions++; return 'Must not run'; })], toolOutputRegistry: new CheckpointRegistry(), toolResultProtection: policy({ inspect: () => { inspections++; return approve('Allowed canonical control'); } }) });
  await expect(node.invoke(state(), { configurable: { run_id: `hidden-checkpoint-${field}` } })).rejects.toMatchObject({ code: 'unsupported' }); expect(inspections).toBe(0); expect(executions).toBe(0);
});

it.each([false, true])('rejects selected host symbol aliases before completion (eager=%s)', async (eager) => {
  const completions: string[] = []; let inspections = 0;
  const run = await Run.create({ runId: `symbol-host-${eager}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, instructions: 'Allowed control.', toolDefinitions: [{ name: 'lookup', parameters: { type: 'object', properties: { count: { type: 'number' } } } }] }, toolResultProtection: policy({ inspect: () => { inspections++; return approve('Allowed canonical control'); } }), eagerEventToolExecution: { enabled: eager }, customHandlers: {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_TOOL_EXECUTE]: { handle: (_event, data): void => { const request = data as ToolExecuteBatchRequest; const result: ToolExecuteResult = { toolCallId: 'call-control', status: 'success', content: 'Allowed control' }; Object.defineProperty(result, Symbol('raw-alias'), { value: fixtures.canaries[0], enumerable: true }); request.resolve([result]); } },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } },
  }, skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: [{ id: 'call-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await expect(run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `symbol-host-${eager}` } })).rejects.toMatchObject({ code: 'unsupported' }); expect(inspections).toBe(0); expect(completions).toEqual([]); assertNoCanary(run.Graph!.getRunMessages());
});

it.each(['content', 'id', 'status', 'metadata'] as const)('revalidates selected nested %s after native callbacks before result extraction', async (field) => {
  let callbacks = 0;
  const lookup = direct(() => fixtures.canaries[0]);
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => {
    callbacks++;
    if (!(output instanceof ToolMessage)) throw new Error('Expected canonical ToolMessage');
    expect(output.content).toBe('[EMAIL_1]');
    if (field === 'content') output.content = fixtures.canaries[0];
    if (field === 'id') output.id = fixtures.canaries[0];
    if (field === 'status') output.status = 'error';
    if (field === 'metadata') output.additional_kwargs._refScope = fixtures.canaries[0];
  } }); callback.awaitHandlers = true; lookup.callbacks = [callback];
  await expect(executeTools([{ id: 'call-control', name: 'lookup', input: { count: 42 } }], new Map([['lookup', lookup]]), 'run_tools_with_code', { policy: policy() })).rejects.toMatchObject({ code: 'unsupported' });
  expect(callbacks).toBe(1);
});

it.each(['ToolResultProtectionError', 'ProviderTextProtectionError'] as const)('reconstructs producer-supplied %s before native error callbacks', async (kind) => {
  const events: string[] = [];
  const error = kind === 'ToolResultProtectionError' ? new ToolResultProtectionError('blocked') : new ProviderTextProtectionError('timeout');
  error.message = fixtures.canaries[0]; Object.defineProperty(error, 'cause', { value: new Error(fixtures.canaries[0]), enumerable: true }); Object.defineProperty(error, 'output', { value: fixtures.canaries[0], enumerable: true });
  const callback = observer(events);
  callback.handleToolError = (observed: Error): void => {
    events.push(observed.message); expect(observed).not.toBe(error); expect(observed.cause).toBeUndefined();
    for (const key of Reflect.ownKeys(observed)) { const value = Object.getOwnPropertyDescriptor(observed, key)?.value; if (typeof value === 'string') assertNoCanary(value); }
  };
  const node = new ToolNode({ trace: true, tools: [direct(() => { throw error; })], toolResultProtection: policy() });
  const failure = await node.invoke(state(), { callbacks: [callback] }).catch((value: Error) => value);
  expect(failure).toMatchObject({ code: kind === 'ToolResultProtectionError' ? 'blocked' : 'timeout' }); expect(failure).not.toBe(error); assertNoCanary(String(failure)); assertNoCanary(events.join(''));
});

it.each(['hook', 'inspection'] as const)('isolates callback-held post-hook source mutations before replacement (%s)', async (phase) => {
  let retained: ToolMessage | undefined; let executions = 0; const events: string[] = [];
  const hooks = new HookRegistry(); hooks.register('PostToolUse', { hooks: [async () => {
    if (retained == null) throw new Error('Expected canonical message');
    if (phase === 'hook') retained.id = fixtures.canaries[0];
    return { updatedOutput: 'Allowed replacement control' };
  }] });
  const callback = observer(events); const observe = callback.handleToolEnd!.bind(callback);
  callback.handleToolEnd = async (output, ...args): Promise<void> => { await observe(output, ...args); if (!(output instanceof ToolMessage)) throw new Error('Expected canonical message'); retained = output; };
  const gate = policy({ inspect: ({ content }) => {
    if (content === 'Allowed replacement control' && phase === 'inspection') { if (retained == null) throw new Error('Expected canonical message'); retained.id = fixtures.canaries[0]; }
    return approve(content);
  } });
  const node = new ToolNode({ trace: true, tools: [direct(() => { executions++; return 'Allowed source control'; })], hookRegistry: hooks, toolResultProtection: gate, toolOutputReferences: { enabled: true }, toolCallStepIds: new Map([['call-control', 'step-control']]) });
  const result = await node.invoke(state(), { configurable: { run_id: `post-envelope-${phase}` }, callbacks: [callback] }) as { messages: ToolMessage[] };
  expect(executions).toBe(1); expect(result.messages[0].content).toBe('Allowed replacement control'); expect(result.messages[0]).not.toBe(retained); assertNoCanary(result); assertNoCanary(events.join(''));
});

it.each(([false, true] as const).flatMap((eager) => (['root-throw', 'root-reject', 'child-throw'] as const).map((path) => [eager, path] as const)))('keeps externally supplied policy exceptions raw-free across required host dispatch (%s/%s)', async (eager, path) => {
  const forged = new ToolResultProtectionError('blocked'); forged.message = fixtures.canaries[0]; Object.defineProperty(forged, 'cause', { value: new Error(fixtures.canaries[0]) }); Object.defineProperty(forged, 'output', { value: fixtures.canaries[0], enumerable: true });
  const updates: string[] = [];
  const toolDefinitions = [{ name: 'lookup', parameters: { type: 'object' as const, properties: { count: { type: 'number' as const } } } }];
  const graphConfig = path === 'child-throw'
    ? { type: 'standard' as const, agents: [{ agentId: 'parent', provider: Providers.OPENAI, instructions: 'Allowed parent.', subagentConfigs: [{ type: 'worker', name: 'Worker', description: 'Allowed child.', agentInputs: { agentId: 'child', provider: Providers.OPENAI, instructions: 'Allowed child.', toolDefinitions } }] }] }
    : { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, instructions: 'Allowed root.', toolDefinitions };
  const run = await Run.create({ runId: `forged-host-${eager}-${path}`, graphConfig, toolResultProtection: policy(), eagerEventToolExecution: { enabled: eager }, customHandlers: {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_TOOL_EXECUTE]: { handle: (_event, data): void => { if (path === 'root-reject') { (data as ToolExecuteBatchRequest).reject(forged); return; } throw forged; } },
    [GraphEvents.ON_SUBAGENT_UPDATE]: { handle: (_event, data): void => { updates.push(JSON.stringify(data)); } },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { updates.push(JSON.stringify(data)); } },
  }, skipCleanup: true });
  const lookupCalls = [{ id: 'call-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' as const }];
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: path === 'child-throw' ? [{ id: 'child-control', name: Constants.SUBAGENT, args: { description: 'Allowed task.', subagent_type: 'worker' }, type: 'tool_call' }] : lookupCalls });
  if (path === 'child-throw') run.Graph!.setSubagentModelOverride(new FakeChatModel({ responses: ['', 'Must not continue'], toolCalls: lookupCalls }));
  const error = await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `forged-host-${eager}-${path}` } }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'blocked' }); expect(error).not.toBe(forged); expect((error as Error).cause).toBeUndefined(); assertNoCanary(String(error)); assertNoCanary(updates.join('')); assertNoCanary(run.Graph!.getRunMessages());
});

it.each(['local', 'python', 'bash'] as const)('blocks callback-mutated nested results before the %s runner can reuse them', async (runtime) => {
  const requests: string[] = [];
  const server = createServer((request, response) => {
    let body = ''; request.on('data', (chunk: Buffer): void => { body += chunk.toString(); });
    request.on('end', (): void => { requests.push(body); response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify({ status: 'tool_call_required', continuation_token: 'Allowed continuation', tool_calls: [{ id: 'call-control', name: 'lookup', input: { count: 42 } }] })); });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { if (!(output instanceof ToolMessage)) throw new Error('Expected canonical message'); expect(output.content).toBe('[EMAIL_1]'); output.content = fixtures.canaries[0]; } }); callback.awaitHandlers = true;
    const lookup = direct(() => fixtures.canaries[0]); lookup.callbacks = [callback];
    let runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
    if (runtime === 'python') runner = createProgrammaticToolCallingTool({ baseUrl });
    if (runtime === 'bash') runner = createBashProgrammaticToolCallingTool({ baseUrl });
    await expect(runner.invoke({ lang: 'bash', code: runtime === 'python' ? 'print(await lookup(count=42))' : 'lookup \'{"count":42}\' || true', tool_manifest: ['lookup'] }, { toolCall: { id: 'nested-control', name: runner.name, type: 'tool_call', args: {}, toolMap: new Map([['lookup', lookup]]), toolDefs: [{ name: 'lookup', allowed_callers: ['code_execution'], parameters: { type: 'object', properties: { count: { type: 'number' } } } }], toolResultProtection: policy() } })).rejects.toMatchObject({ code: 'unsupported' });
    expect(requests).toHaveLength(runtime === 'local' ? 0 : 1); assertNoCanary(requests);
  } finally { server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve())); }
});

it('normalizes a producer protection error without invoking its code accessor', async () => {
  const forged = new ToolResultProtectionError('blocked'); let reads = 0; Object.defineProperty(forged, 'code', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } }); forged.message = fixtures.canaries[0];
  const events: string[] = [];
  const node = new ToolNode({ trace: true, tools: [direct(() => { throw forged; })], toolResultProtection: policy() });
  const error = await node.invoke(state(), { callbacks: [observer(events)] }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'incompatible' }); expect(reads).toBe(0); assertNoCanary(events.join('')); assertNoCanary(String(error));
});

it('preserves absent-policy nested exception behavior', async () => {
  const original = new ProviderTextProtectionError('blocked');
  await expect(executeTools([{ id: 'plain-control', name: 'lookup', input: { count: 42 } }], new Map([['lookup', direct(() => { throw original; })]]))).rejects.toBe(original);
});

it.each([false, true])('never logs handler-mutated selected errors before required reinspection (block=%s)', async (block) => {
  const logs: string[] = []; const inspected: string[] = [];
  jest.spyOn(console, 'error').mockImplementation((...args): void => { logs.push(JSON.stringify(args)); });
  const node = new ToolNode({ tools: [direct(() => { throw new Error('Allowed initial error'); })], toolResultProtection: policy({ inspect: ({ content }) => { inspected.push(content); return block && content.includes(fixtures.canaries[0]) ? { version: 1, ok: false, error: { code: 'blocked' } } : approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } }), errorHandler: async ({ error }) => {
    error!.message = fixtures.canaries[0]; error!.stack = fixtures.canaries[0];
    throw new Error(fixtures.canaries[0], { cause: error });
  } });
  const pending = node.invoke(state());
  if (block) await expect(pending).rejects.toMatchObject({ code: 'blocked' });
  else assertNoCanary(await pending);
  expect(inspected).toEqual(['Allowed initial error', fixtures.canaries[0]]);
  expect(logs).toHaveLength(1); assertNoCanary(logs.join(''));
});

it.each(['lookup', 'unregistered'] as const)('rejects required client delegation before graph execution (%s)', async (name) => {
  let inspections = 0;
  await expect(Run.create({ runId: `required-delegation-${name}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [direct(() => 'Must not execute')] }, clientDelegatedToolNames: [name], toolResultProtection: policy({ toolNames: [name], inspect: () => { inspections++; return approve('Must not inspect'); } }), customHandlers: { [GraphEvents.ON_MODEL_RESPONSE]: { handle: (): void => {} } } })).rejects.toMatchObject({ code: 'unsupported' });
  expect(inspections).toBe(0);
});

it.each([false, true])('preserves unselected or absent-policy client delegation (configured=%s)', async (configured) => {
  const accepted: string[] = []; let executions = 0;
  const run = await Run.create({ runId: `allowed-delegation-${configured}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [direct(() => { executions++; return 'Must not execute on server'; })] }, clientDelegatedToolNames: ['lookup'], toolResultProtection: configured ? policy({ toolNames: ['other'] }) : undefined, customHandlers: { [GraphEvents.ON_MODEL_RESPONSE]: { handle: (_event, data): void => { accepted.push(JSON.stringify(data)); } } }, skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'delegated-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `allowed-delegation-${configured}` } });
  expect(executions).toBe(0); expect(accepted.join('')).toContain('delegated-control');
});

it.each(['content', 'id', 'metadata'] as const)('detaches native callback-held %s before the next model request', async (field) => {
  let retained: ToolMessage | undefined; const inputs: string[] = []; let executions = 0;
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { if (!(output instanceof ToolMessage)) throw new Error('Expected canonical message'); retained = output; } }); callback.awaitHandlers = true;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; }); lookup.callbacks = [callback];
  class Model extends FakeChatModel {
    override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> {
      if (retained != null) {
        const alias = retained;
        await new Promise<void>((resolve) => setImmediate(() => {
          if (field === 'content') alias.content = fixtures.canaries[0];
          if (field === 'id') alias.id = fixtures.canaries[0];
          if (field === 'metadata') alias.additional_kwargs._refScope = fixtures.canaries[0];
          resolve();
        }));
      }
      inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args);
    }
  }
  const run = await Run.create({ runId: `callback-detach-${field}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup] }, toolResultProtection: policy(), toolOutputReferences: { enabled: true }, skipCleanup: true });
  run.Graph!.overrideModel = new Model({ responses: ['', 'Allowed final answer'], toolCalls: [{ id: 'call-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `callback-detach-${field}` } });
  expect(executions).toBe(1); expect(retained).toBeDefined(); expect(inputs).toHaveLength(2); expect(inputs[1]).toContain('[EMAIL_1]'); assertNoCanary(inputs[1]); assertNoCanary(run.Graph!.getRunMessages());
});

it.each(['content', 'metadata'] as const)('returns state detached from a callback-retained %s alias', async (field) => {
  let retained: ToolMessage | undefined;
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { if (!(output instanceof ToolMessage)) throw new Error('Expected canonical message'); retained = output; } }); callback.awaitHandlers = true;
  const node = new ToolNode({ trace: true, tools: [direct(() => fixtures.canaries[0])], toolResultProtection: policy() });
  const result = await node.invoke(state(), { callbacks: [callback] }) as { messages: ToolMessage[] };
  expect(retained).toBeDefined();
  if (retained == null) throw new Error('Expected callback retention');
  await new Promise<void>((resolve) => setImmediate(() => {
    if (field === 'content') retained!.content = fixtures.canaries[0];
    if (field === 'metadata') retained!.additional_kwargs._refScope = fixtures.canaries[0];
    resolve();
  }));
  expect(result.messages[0]).not.toBe(retained); expect(result.messages[0].additional_kwargs).not.toBe(retained.additional_kwargs);
  expect(result.messages[0].content).toBe('[EMAIL_1]'); assertNoCanary(result);
});

it('rejects required delegation through the standalone graph constructor', () => {
  expect(() => new StandardGraph({ runId: 'direct-required-delegation', agents: [{ agentId: 'agent', provider: Providers.OPENAI }], clientDelegatedToolNames: ['lookup'], toolResultProtection: policy() })).toThrow(ToolResultProtectionError);
});

it('preserves ordinary handler diagnostics when result protection is absent', async () => {
  const logs: string[] = []; jest.spyOn(console, 'error').mockImplementation((...args): void => { logs.push(JSON.stringify(args)); });
  const node = new ToolNode({ tools: [direct(() => { throw new Error('Allowed original control'); })], errorHandler: async () => { throw new Error('Allowed observer failure'); } });
  const result = await node.invoke(state()); expect(JSON.stringify(result)).toContain('Allowed original control'); expect(logs.join('')).toContain('Allowed observer failure');
});

it.each(['legacy-id', 'legacy-scope', 'legacy-safe', 'protected', 'protected-refs', 'stripped', 'tampered-id', 'tampered-scope', 'future'] as const)('requires certified envelope aliases on settled checkpoint replay (%s)', async (kind) => {
  let executions = 0; let approvals = 0; const inspected: string[] = []; const events: string[] = [];
  const lookup = direct(() => {
    executions++;
    return new ToolMessage({ id: kind === 'legacy-id' ? fixtures.canaries[0] : 'call-control', name: 'lookup', tool_call_id: 'call-control', status: 'success', content: fixtures.canaries[0], additional_kwargs: kind === 'legacy-scope' ? { _refScope: fixtures.canaries[0] } : {} });
  });
  const approval = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed approval control.', schema: z.object({ count: z.number() }), func: async () => { approvals++; return 'Approved control'; } });
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'approval_control' ? 'ask' : 'allow', reason: 'Allowed approval' })] });
  const gate = policy({ inspect: ({ content }) => { inspected.push(content); return approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } });
  const saver = new MemorySaver();
  const certified = !kind.startsWith('legacy');
  const references = certified && kind !== 'protected';
  const compile = (protectedRun: boolean): Pick<Runnable<unknown, unknown>, 'invoke'> => new StateGraph(MessagesAnnotation).addNode('tools', new ToolNode({ trace: true, tools: [lookup, approval], hookRegistry: hooks, humanInTheLoop: { enabled: true }, toolResultProtection: protectedRun ? gate : undefined, toolOutputReferences: { enabled: references } })).addEdge(START, 'tools').addEdge('tools', END).compile({ checkpointer: saver });
  const config = { configurable: { thread_id: `envelope-replay-${kind}`, run_id: `envelope-replay-${kind}` }, callbacks: [observer(events)] };
  const paused = await compile(certified).invoke({ messages: [new AIMessage({ id: 'envelope-batch', content: '', tool_calls: [{ id: 'call-control', name: 'lookup', args: { count: 42 } }, { id: 'approval-control', name: 'approval_control', args: { count: 42 } }] })] }, config);
  expect(isInterrupted(paused)).toBe(true); if (!isInterrupted(paused)) throw new Error('Expected approval checkpoint');
  const replay = getToolBatchReplayState(paused.__interrupt__[0].value); expect(replay).toBeDefined(); expect(executions).toBe(1); expect(approvals).toBe(0); events.length = 0;
  if (['stripped', 'tampered-id', 'tampered-scope', 'future'].includes(kind)) {
    if (replay == null) throw new Error('Expected replay state');
    const record = replay.records[0];
    const restored = await saver.serde.loadsTyped(record.encoding, Buffer.from(record.data, 'base64')) as Array<[string, SettledToolBatchResult]>;
    const settled = restored[0][1];
    if (!(settled.output instanceof ToolMessage)) throw new Error('Expected settled ToolMessage');
    if (kind === 'stripped') delete settled.release;
    if (kind === 'tampered-id') settled.output.id = fixtures.canaries[0];
    if (kind === 'tampered-scope') settled.output.additional_kwargs._refScope = fixtures.canaries[0];
    if (kind === 'future') Object.defineProperty(settled.release!, 'version', { value: 2 });
    const [encoding, bytes] = await saver.serde.dumpsTyped(restored); record.encoding = encoding; record.data = Buffer.from(bytes).toString('base64');
  }
  const resumed = compile(true).invoke(new Command({ resume: [{ type: 'approve' }] }), { ...config, configurable: { ...config.configurable, [TOOL_BATCH_REPLAY_KEY]: replay } });
  if (['legacy-id', 'legacy-scope', 'stripped', 'tampered-id', 'tampered-scope', 'future'].includes(kind)) await expect(resumed).rejects.toMatchObject({ code: kind === 'future' ? 'incompatible' : 'unsupported' });
  else { const result = await resumed; assertNoCanary(result); expect(JSON.stringify(result)).toContain('[EMAIL_1]'); expect(approvals).toBe(1); expect(inspected).toEqual(certified ? [fixtures.canaries[0], '[EMAIL_1]'] : [fixtures.canaries[0]]); }
  expect(executions).toBe(1); assertNoCanary(events.join(''));
});

it.each(['stack', 'cause', 'alias', 'message', 'accessor'] as const)('detaches released %s errors after native callbacks when tool errors are not handled', async (field) => {
  let retained: Error | undefined; const inspected: string[] = [];
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error: Error): void => {
    expect(error.message).toBe('[EMAIL_1]'); retained = error;
    if (field === 'stack') error.stack = fixtures.canaries[0];
    if (field === 'cause') Object.defineProperty(error, 'cause', { value: new Error(fixtures.canaries[0]), enumerable: true });
    if (field === 'alias') Object.defineProperty(error, Symbol('raw-alias'), { value: fixtures.canaries[0], enumerable: true });
    if (field === 'message') error.message = fixtures.canaries[0];
    if (field === 'accessor') Object.defineProperty(error, 'message', { get: () => { throw new Error(fixtures.canaries[0]); } });
  } }); callback.awaitHandlers = true;
  const node = new ToolNode({ trace: true, tools: [direct(() => { throw new Error(fixtures.canaries[0]); })], handleToolErrors: false, toolResultProtection: policy({ inspect: ({ content }) => { inspected.push(content); return approve('[EMAIL_1]'); } }) });
  const error = await node.invoke(state(), { callbacks: [callback] }).catch((value: Error) => value);
  expect(error).toBeInstanceOf(Error); expect(error).not.toBe(retained);
  if (field === 'message' || field === 'accessor') expect(error).toMatchObject({ code: 'unsupported' });
  else expect(error).toMatchObject({ message: '[EMAIL_1]' }); expect((error as Error).cause).toBeUndefined();
  for (const key of Reflect.ownKeys(error as Error)) { const value = Object.getOwnPropertyDescriptor(error, key)?.value; if (typeof value === 'string') assertNoCanary(value); }
  expect(inspected).toEqual([fixtures.canaries[0]]);
});

it.each(['legacy', 'protected', 'unselected', 'stricter', 'blocked'] as const)('gates earlier committed checkpoint ToolMessages before a rebuilt Run resumes (%s)', async (mode) => {
  let executions = 0; let approvals = 0; const resumedInputs: string[] = []; const inspections: string[] = [];
  const sourceName = mode === 'unselected' ? 'other' : 'lookup';
  const source = new DynamicStructuredTool({ name: sourceName, description: 'Allowed source control.', schema: z.object({ count: z.number() }), func: async () => { executions++; return mode === 'unselected' ? 'Allowed unselected control' : fixtures.canaries[0]; } });
  const approval = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed approval control.', schema: z.object({ count: z.number() }), func: async () => { approvals++; return 'Approved control'; } });
  class Model extends FakeChatModel {
    constructor(private readonly resumed: boolean) { super({ responses: ['Allowed final answer'] }); }
    override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> {
      if (this.resumed) resumedInputs.push(JSON.stringify(args[0]));
      const tools = args[0].filter((message) => message.getType() === 'tool') as ToolMessage[];
      const next = !tools.some((message) => message.tool_call_id === 'historical-source') ? { id: 'historical-source', name: sourceName, args: { count: 42 }, type: 'tool_call' as const } : undefined;
      const call = next ?? (!tools.some((message) => message.tool_call_id === 'historical-approval') ? { id: 'historical-approval', name: 'approval_control', args: { count: 42 }, type: 'tool_call' as const } : undefined);
      yield* new FakeChatModel({ responses: [call == null ? 'Allowed final answer' : ''], toolCalls: call == null ? [] : [call] })._streamResponseChunks(...args);
    }
  }
  const saver = new MemorySaver(); const runId = `committed-history-${mode}`;
  const create = async (gate: ToolResultProtection | undefined, resumed: boolean): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'approval_control' ? 'ask' : 'allow', reason: 'Allowed approval' })] });
    const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [source, approval], compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: gate, skipCleanup: true });
    run.Graph!.overrideModel = new Model(resumed); return run;
  };
  let previous: ToolResultProtection | undefined = policy();
  if (mode === 'legacy') previous = undefined;
  if (mode === 'stricter') previous = policy({ inspect: ({ content }) => approve(content) });
  const first = await create(previous, false);
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await first.processStream({ messages: [new HumanMessage('Allowed control')] }, config);
  expect(first.getInterrupt()).toBeDefined(); expect(executions).toBe(1); expect(approvals).toBe(0);
  const gate = policy({ inspect: ({ content }) => { inspections.push(content); return mode === 'blocked' ? { version: 1, ok: false, error: { code: 'blocked' } } : approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } });
  const resumed = await create(gate, true);
  const pending = resumed.resume([{ type: 'approve' }], config);
  if (mode === 'legacy' || mode === 'stricter' || mode === 'blocked') {
    await expect(pending).rejects.toMatchObject({ code: mode === 'blocked' ? 'blocked' : 'incompatible' }); expect(resumedInputs).toEqual([]); expect(approvals).toBe(0);
  } else {
    await pending; expect(resumedInputs).toHaveLength(1); assertNoCanary(resumedInputs[0]); expect(resumedInputs[0]).toContain(mode === 'unselected' ? 'Allowed unselected control' : '[EMAIL_1]'); expect(approvals).toBe(1);
  }
  expect(executions).toBe(1);
});

it.each(['legacy', 'protected'] as const)('gates committed child history before executing a resumed child tool (%s)', async (mode) => {
  let executions = 0; let approvals = 0; const resumedInputs: string[] = [];
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const approval = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed child approval.', schema: z.object({ count: z.number() }), func: async () => { approvals++; return 'Approved child control'; } });
  class ChildModel extends FakeChatModel {
    constructor(private readonly resumed: boolean) { super({ responses: ['Allowed child answer'] }); }
    override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> {
      if (this.resumed) resumedInputs.push(JSON.stringify(args[0]));
      const tools = args[0].filter((message) => message.getType() === 'tool') as ToolMessage[];
      let call: ToolCall | undefined;
      if (!tools.some((message) => message.tool_call_id === 'child-source')) call = { id: 'child-source', name: 'lookup', args: { count: 42 }, type: 'tool_call' };
      else if (!tools.some((message) => message.tool_call_id === 'child-approval')) call = { id: 'child-approval', name: 'approval_control', args: { count: 42 }, type: 'tool_call' };
      yield* new FakeChatModel({ responses: [call == null ? 'Allowed child answer' : ''], toolCalls: call == null ? [] : [call] })._streamResponseChunks(...args);
    }
  }
  const runId = `committed-child-${mode}`; const saver = new MemorySaver();
  const create = async (protectedRun: boolean, resumed: boolean): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'approval_control' ? 'ask' : 'allow', reason: 'Allowed child approval' })] });
    const run = await Run.create({ runId, graphConfig: { type: 'standard', agents: [{ agentId: 'parent', provider: Providers.OPENAI, instructions: 'Allowed parent.', subagentConfigs: [{ type: 'worker', name: 'Worker', description: 'Allowed child.', agentInputs: { agentId: 'child', provider: Providers.OPENAI, instructions: 'Allowed child.', tools: [lookup, approval] } }] }], compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: protectedRun ? policy() : undefined, skipCleanup: true });
    run.Graph!.overrideModel = new FakeChatModel({ responses: resumed ? ['Allowed final answer'] : ['', 'Allowed final answer'], toolCalls: resumed ? [] : [{ id: 'worker-control', name: Constants.SUBAGENT, args: { description: 'Allowed child task.', subagent_type: 'worker' }, type: 'tool_call' }] });
    run.Graph!.setSubagentModelOverride(new ChildModel(resumed)); return run;
  };
  const first = await create(mode === 'protected', false); const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await first.processStream({ messages: [new HumanMessage('Allowed control')] }, config); expect(first.getInterrupt()).toBeDefined(); expect(executions).toBe(1); expect(approvals).toBe(0);
  const resumed = await create(true, true); const pending = resumed.resume([{ type: 'approve' }], config);
  if (mode === 'legacy') { await expect(pending).rejects.toMatchObject({ code: 'incompatible' }); expect(approvals).toBe(0); expect(resumedInputs).toEqual([]); }
  else { await pending; expect(approvals).toBe(1); assertNoCanary(resumedInputs.join('')); assertNoCanary(resumed.Graph!.getRunMessages()); }
  expect(executions).toBe(1);
});

it.each((['same-run', 'rebuilt', 'legacy', 'unselected', 'stricter'] as const).flatMap((mode) => [false, true].map((toolEnd) => [mode, toolEnd] as const)))('preserves certified checkpoint history on a fresh same-thread turn (%s/toolEnd=%s)', async (mode, toolEnd) => {
  const inputs: string[] = []; let executions = 0;
  const name = mode === 'unselected' ? 'other' : 'lookup';
  const lookup = new DynamicStructuredTool({ name, description: 'Allowed source control.', schema: z.object({ count: z.number() }), func: async () => { executions++; return mode === 'unselected' ? 'Allowed unselected control' : fixtures.canaries[0]; } });
  const saver = new MemorySaver(); const runId = `fresh-history-${mode}-${toolEnd}`;
  const create = (protectedRun: boolean, gate = policy()): Promise<Run<IState>> => Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd, compileOptions: { checkpointer: saver } }, toolResultProtection: protectedRun ? gate : undefined });
  const first = await create(mode !== 'legacy', mode === 'stricter' ? policy({ inspect: ({ content }) => approve(content) }) : policy()); first.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed first answer'], toolCalls: [{ id: 'first-source', name, args: { count: 42 }, type: 'tool_call' }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await first.processStream({ messages: [new HumanMessage('Allowed first control')] }, config); expect(executions).toBe(1);
  const workflow = first.graphRunnable as typeof first.graphRunnable & { getState(config: RunnableConfig): Promise<{ values: { messages: BaseMessage[]; runStepState: RunStepResumeState } }> };
  const committed = await workflow.getState(config);
  const committedTool = committed.values.messages.find((message: BaseMessage) => message instanceof ToolMessage && message.tool_call_id === 'first-source') as ToolMessage;
  expect(committedTool.id).toEqual(expect.any(String));
  if (mode !== 'legacy' && mode !== 'unselected') expect(committed.values.runStepState.toolResultReleases).toEqual(expect.arrayContaining([expect.objectContaining({ id: 'first-source', envelope: expect.objectContaining({ id: committedTool.id }) })]));
  const next = mode === 'same-run' ? first : await create(true);
  class Model extends FakeChatModel {
    override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); }
  }
  next.Graph!.overrideModel = new Model({ responses: ['Allowed next answer'] });
  const pending = next.processStream({ messages: [new HumanMessage('Allowed next control')] }, config);
  if (mode === 'legacy' || mode === 'stricter') { await expect(pending).rejects.toMatchObject({ code: 'incompatible' }); expect(inputs).toEqual([]); }
  else { await pending; expect(inputs).toHaveLength(1); expect(inputs[0]).toContain(mode === 'unselected' ? 'Allowed unselected control' : '[EMAIL_1]'); assertNoCanary(inputs[0]); }
  expect(executions).toBe(1);
});

it.each((['direct', 'host', 'eager'] as const).flatMap((path) => (['success', 'error'] as const).map((status) => [path, status] as const)))('atomically checkpoints terminal %s/%s results and receipts before the next turn', async (path, status) => {
  let executions = 0; const inputs: string[] = []; const saver = new MemorySaver(); const runId = `terminal-${path}-${status}`;
  const lookup = direct(() => { executions++; if (status === 'error') throw new Error(fixtures.canaries[0]); return fixtures.canaries[0]; });
  const create = (): Promise<Run<IState>> => Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: path === 'direct' ? [lookup] : undefined, toolDefinitions: path === 'direct' ? undefined : [{ name: 'lookup', parameters: { type: 'object', properties: { count: { type: 'number' } } } }], toolEnd: true, compileOptions: { checkpointer: saver } }, toolResultProtection: policy(), eagerEventToolExecution: { enabled: path === 'eager' }, customHandlers: {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_TOOL_EXECUTE]: { handle: (_event, data): void => { executions++; const request = data as ToolExecuteBatchRequest; request.resolve([{ toolCallId: 'terminal-source', status, content: status === 'success' ? fixtures.canaries[0] : '', ...(status === 'error' ? { errorMessage: fixtures.canaries[0] } : {}) }]); } },
  } });
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'terminal-source', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await first.processStream({ messages: [new HumanMessage('Allowed first control')] }, config);
  const workflow = first.graphRunnable as typeof first.graphRunnable & { getState(config: RunnableConfig): Promise<{ values: { messages: BaseMessage[]; runStepState: RunStepResumeState } }> };
  const checkpoint = await workflow.getState(config);
  const message = checkpoint.values.messages.find((entry) => entry instanceof ToolMessage && entry.tool_call_id === 'terminal-source') as ToolMessage;
  expect(message.status).toBe(status); assertNoCanary(message); expect(checkpoint.values.runStepState.toolResultReleases).toEqual(expect.arrayContaining([expect.objectContaining({ id: 'terminal-source', envelope: expect.objectContaining({ id: message.id, status }) })]));
  const resumed = await create();
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  resumed.Graph!.overrideModel = new Model({ responses: ['Allowed next answer'] });
  await resumed.processStream({ messages: [new HumanMessage('Allowed next control')] }, config);
  expect(executions).toBe(1); expect(inputs).toHaveLength(1); expect(inputs[0]).toContain('[EMAIL_1]'); assertNoCanary(inputs[0]);
});

it.each(['accessor', 'message', 'stack', 'cause', 'alias', 'unchanged'] as const)('validates nested post-callback error %s before extraction', async (field) => {
  let reads = 0; const inspected: string[] = []; let callbacks = 0;
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error: Error): void => {
    callbacks++; expect(error.message).toBe('[EMAIL_1]');
    if (field === 'accessor') Object.defineProperty(error, 'message', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
    if (field === 'message') error.message = fixtures.canaries[0];
    if (field === 'stack') error.stack = fixtures.canaries[0];
    if (field === 'cause') Object.defineProperty(error, 'cause', { value: new Error(fixtures.canaries[0]) });
    if (field === 'alias') Object.defineProperty(error, Symbol('raw-alias'), { value: fixtures.canaries[0] });
  } }); callback.awaitHandlers = true;
  const lookup = direct(() => { throw new Error(fixtures.canaries[0]); }); lookup.callbacks = [callback];
  const pending = executeTools([{ id: 'nested-error', name: 'lookup', input: { count: 42 } }], new Map([['lookup', lookup]]), 'run_tools_with_code', { policy: policy({ inspect: ({ content }) => { inspected.push(content); return approve('[EMAIL_1]'); } }) });
  if (field === 'accessor' || field === 'message') await expect(pending).rejects.toMatchObject({ code: 'unsupported' });
  else { const result = await pending; expect(result).toEqual([{ call_id: 'nested-error', is_error: true, result: null, error_message: '[EMAIL_1]' }]); assertNoCanary(result); }
  expect(reads).toBe(0); expect(callbacks).toBe(1); expect(inspected).toEqual([fixtures.canaries[0]]);
});

it.each(['local', 'python', 'bash'] as const)('keeps callback-installed error accessors terminal before %s runner reuse', async (runtime) => {
  const requests: string[] = []; let reads = 0; let callbacks = 0;
  const server = createServer((request, response) => {
    let body = ''; request.on('data', (chunk: Buffer): void => { body += chunk.toString(); });
    request.on('end', (): void => { requests.push(body); response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(requests.length === 1 ? { status: 'tool_call_required', continuation_token: 'Allowed continuation', tool_calls: [{ id: 'nested-error', name: 'lookup', input: { count: 42 } }] } : { status: 'completed', stdout: 'Must not continue', files: [] })); });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error: Error): void => { callbacks++; expect(error.message).toBe('[EMAIL_1]'); Object.defineProperty(error, 'message', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } }); } }); callback.awaitHandlers = true;
    const lookup = direct(() => { throw new Error(fixtures.canaries[0]); }); lookup.callbacks = [callback];
    const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    let runner = createLocalProgrammaticToolCallingTool({ cwd: process.cwd() });
    if (runtime === 'python') runner = createProgrammaticToolCallingTool({ baseUrl });
    if (runtime === 'bash') runner = createBashProgrammaticToolCallingTool({ baseUrl });
    const events: string[] = [];
    const error = await runner.invoke({ lang: 'bash', code: runtime === 'python' ? 'print(await lookup(count=42))' : 'lookup \'{"count":42}\' || true', tool_manifest: ['lookup'] }, { callbacks: [observer(events)], toolCall: { id: 'outer-error', name: runner.name, type: 'tool_call', args: {}, toolMap: new Map([['lookup', lookup]]), toolDefs: [{ name: 'lookup', allowed_callers: ['code_execution'], parameters: { type: 'object', properties: { count: { type: 'number' } } } }], toolResultProtection: policy() } }).catch((value: Error) => value);
    expect(error).toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0); expect(callbacks).toBe(1); expect(requests).toHaveLength(runtime === 'local' ? 0 : 1); assertNoCanary(String(error)); assertNoCanary(events.join('')); assertNoCanary(requests);
  } finally { server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve())); }
});

it.each((['changing', 'throwing', 'replacement', 'stack', 'name'] as const).flatMap((kind) => [false, true].map((throws) => [kind, throws] as const)))('snapshots selected post-handler %s error text without accessor reads (throws=%s)', async (kind, throws) => {
  let reads = 0; let handlers = 0; const inspected: string[] = []; const logs: string[] = []; const events: string[] = [];
  jest.spyOn(console, 'error').mockImplementation((...args): void => { logs.push(JSON.stringify(args)); });
  const node = new ToolNode({ trace: true, tools: [direct(() => { throw new Error(fixtures.canaries[0]); })], toolCallStepIds: new Map([['call-control', 'step-control']]), toolOutputReferences: { enabled: true }, toolResultProtection: policy({ inspect: ({ content }) => { inspected.push(content); return approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } }), errorHandler: async ({ error }) => {
    handlers++; expect(error!.message).toBe('[EMAIL_1]');
    if (kind === 'changing') Object.defineProperty(error, 'message', { get: () => { reads++; return reads === 1 ? '[EMAIL_1]' : fixtures.canaries[0]; } });
    if (kind === 'throwing') Object.defineProperty(error, 'message', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
    if (kind === 'replacement') error!.message = fixtures.canaries[0];
    if (kind === 'stack') { error!.stack = fixtures.canaries[0]; Object.defineProperty(error, 'cause', { value: new Error(fixtures.canaries[0]) }); }
    if (kind === 'name') Object.defineProperty(error, 'name', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
    if (throws) throw new Error(fixtures.canaries[0]);
    return false;
  } });
  const pending = node.invoke(state(), { configurable: { run_id: `post-error-${kind}-${throws}` }, callbacks: [observer(events)] });
  if (kind === 'changing' || kind === 'throwing') await expect(pending).rejects.toMatchObject({ code: 'unsupported' });
  else { const output = await pending; assertNoCanary(output); expect(JSON.stringify(output)).toContain('[EMAIL_1]'); }
  expect(reads).toBe(0); expect(handlers).toBe(1); expect(inspected).toEqual(kind === 'replacement' ? [fixtures.canaries[0], fixtures.canaries[0]] : [fixtures.canaries[0]]); assertNoCanary(events.join('')); assertNoCanary(logs.join('')); expect(node._unsafeGetToolOutputRegistry()!.size).toBe(0);
});

it('uses the detached post-handler data snapshot across an awaited replacement inspection', async () => {
  const started = deferred<void>(); const decision = deferred<ToolResultProtectionResult>(); let retained: Error | undefined; const inspected: string[] = []; const events: string[] = [];
  const node = new ToolNode({ trace: true, tools: [direct(() => { throw new Error('Allowed initial error'); })], toolResultProtection: policy({ inspect: ({ content }) => { inspected.push(content); if (content === 'Allowed replacement error') { started.resolve(); return decision.promise; } return approve(content); } }), errorHandler: async ({ error }) => { retained = error; error!.message = 'Allowed replacement error'; return false; } });
  const pending = node.invoke(state(), { callbacks: [observer(events)] });
  await started.promise; retained!.message = fixtures.canaries[0]; retained!.stack = fixtures.canaries[0]; decision.resolve(approve('Allowed canonical replacement'));
  const output = await pending; expect(inspected).toEqual(['Allowed initial error', 'Allowed replacement error']); expect(JSON.stringify(output)).toContain('Allowed canonical replacement'); assertNoCanary(output); assertNoCanary(events.join(''));
});

it('preserves admitted empty error text without a second policy decision', async () => {
  let inspections = 0; const handled: string[] = [];
  const node = new ToolNode({ tools: [direct(() => { throw new Error('Allowed initial error'); })], toolResultProtection: policy({ inspect: () => { inspections++; return approve(''); } }), errorHandler: async ({ error }) => { handled.push(error!.message); return false; } });
  const result = await node.invoke(state()) as { messages: ToolMessage[] };
  expect(inspections).toBe(1); expect(handled).toEqual(['']); expect(result.messages[0].content).toBe('Error: \n Please fix your mistakes.');
});

async function certifiedHistory(): Promise<{ gate: ToolResultProtection; messages: BaseMessage[]; releases: NonNullable<RunStepResumeState['toolResultReleases']> }> {
  const gate = policy();
  const producer = new ToolNode({ tools: [direct(() => fixtures.canaries[0])], toolResultProtection: gate });
  const result = await producer.invoke(state()) as { messages: ToolMessage[] };
  const message = result.messages[0];
  const release = getToolMessageRelease(gate, 'lookup', 'call-control', message)!;
  return { gate, messages: [state().messages[0], message], releases: [release] };
}

it.each(['iterator', 'proxy', 'subclass', 'index', 'sparse'] as const)('rejects checkpoint message collection %s without controlled reads', async (kind) => {
  const fixture = await certifiedHistory(); let reads = 0; let messages = fixture.messages;
  if (kind === 'iterator') Object.defineProperty(messages, Symbol.iterator, { value: function* () { reads++; if (reads > 1) yield* fixture.messages; } });
  if (kind === 'proxy') messages = new Proxy(messages, { get: (target, key, receiver): unknown => { reads++; return Reflect.get(target, key, receiver); } });
  if (kind === 'subclass') { class MessageArray extends Array<BaseMessage> {} messages = new MessageArray(...messages); }
  if (kind === 'index') Object.defineProperty(messages, '1', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
  if (kind === 'sparse') messages = new Array<BaseMessage>(2);
  await expect(validateToolMessageHistory(fixture.gate, messages, fixture.releases)).rejects.toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0);
});

it.each(['content', 'name', 'id', 'status', 'tool_call_id', 'additional_kwargs', 'response_metadata'] as const)('rejects inherited historical %s accessors before reading them', async (field) => {
  const fixture = await certifiedHistory(); const original = fixture.messages[1] as ToolMessage; let reads = 0;
  const message = new ToolMessage({ ...original });
  Reflect.deleteProperty(message, field);
  Object.setPrototypeOf(message, Object.create(ToolMessage.prototype, { [field]: { get: () => { reads++; throw new Error(fixtures.canaries[0]); } } }));
  const error = await validateToolMessageHistory(fixture.gate, [fixture.messages[0], message], fixture.releases).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0); assertNoCanary(String(error));
});

it.each(['call', 'message', 'both'] as const)('binds historical %s names to certified result identities', async (changed) => {
  const fixture = await certifiedHistory(); const call = fixture.messages[0] as AIMessage; const message = fixture.messages[1] as ToolMessage;
  const renamed = new AIMessage({ ...call, tool_calls: call.tool_calls!.map((entry) => ({ ...entry, name: changed === 'message' ? entry.name : 'other' })) });
  const aliased = new ToolMessage({ ...message, name: changed === 'call' ? message.name : 'other', content: fixtures.canaries[0] });
  await expect(validateToolMessageHistory(policy(), [renamed, aliased], fixture.releases)).rejects.toMatchObject({ code: 'unsupported' });
});

it('rejects restored protected references renamed to an unselected source', async () => {
  let executions = 0;
  const registry = new ToolOutputReferenceRegistry(); registry.restoreState('renamed-reference', { entries: [{ key: 'tool0turn0', value: fixtures.canaries[0], protection: { version: 1, toolName: 'other', toolCallId: 'call-control', protected: true } }], turnCounter: 1, warnedNonStringTools: [] });
  const pipe = new DynamicStructuredTool({ name: 'pipe', description: 'Allowed pipe.', schema: z.object({ command: z.string() }), func: async ({ command }) => { executions++; return command; } });
  const node = new ToolNode({ tools: [pipe], toolOutputRegistry: registry, toolResultProtection: policy() });
  await expect(node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'pipe-control', name: 'pipe', args: { command: '{{' + 'tool0turn0' + '}}' } }] })] }, { configurable: { run_id: 'renamed-reference' } })).rejects.toMatchObject({ code: 'incompatible' }); expect(executions).toBe(0);
});

it.each([false, true])('validates supplied tool history before RunStart (checkpoint=%s)', async (checkpoint) => {
  let hooks = 0; let model = 0;
  const registry = new HookRegistry(); registry.register('RunStart', { hooks: [async () => { hooks++; return {}; }] });
  const run = await Run.create({ runId: `supplied-history-${checkpoint}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: checkpoint ? { checkpointer: new MemorySaver() } : undefined }, hooks: registry, toolResultProtection: policy() });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { model++; yield* super._streamResponseChunks(...args); } }
  run.Graph!.overrideModel = new Model({ responses: ['Must not run'] });
  await expect(run.processStream({ messages: [state().messages[0], new ToolMessage({ tool_call_id: 'call-control', name: 'lookup', content: fixtures.canaries[0] }), new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `supplied-history-${checkpoint}` } })).rejects.toMatchObject({ code: 'incompatible' });
  expect(hooks).toBe(0); expect(model).toBe(0);
});

it.each(['remove', 'compact'] as const)('prunes historical release receipts after %s', async (kind) => {
  class Graph extends StandardGraph { reduce(current: BaseMessage[], update: BaseMessage[]): BaseMessage[] { return this.reduceProtectedMessages(current, update); } }
  const fixture = await certifiedHistory(); const graph = new Graph({ runId: `prune-${kind}`, agents: [{ agentId: 'agent', provider: Providers.OPENAI }], toolResultProtection: fixture.gate });
  let messages = graph.reduce([], fixture.messages);
  expect(graph.createRunStepResumeState().toolResultReleases).toHaveLength(1);
  const id = messages[1].id!;
  messages = graph.reduce(messages, kind === 'remove' ? [new RemoveMessage({ id })] : [createRemoveAllMessage(), new HumanMessage('Allowed compacted control')]);
  expect(messages.some((message) => message.id === id)).toBe(false); expect(graph.createRunStepResumeState().toolResultReleases).toEqual([]);
});

it('retains certified results with repeated provider call IDs in distinct message identities', async () => {
  class Graph extends StandardGraph { reduce(current: BaseMessage[], update: BaseMessage[]): BaseMessage[] { return this.reduceProtectedMessages(current, update); } }
  const fixture = await certifiedHistory(); const graph = new Graph({ runId: 'repeated-history', agents: [{ agentId: 'agent', provider: Providers.OPENAI }], toolResultProtection: fixture.gate });
  let history = graph.reduce([], fixture.messages);
  const producer = new ToolNode({ tools: [direct(() => fixtures.canaries[0])], toolResultProtection: fixture.gate });
  const next = await producer.invoke(state()) as { messages: ToolMessage[] };
  history = graph.reduce(history, [new AIMessage({ content: '', tool_calls: [{ id: 'call-control', name: 'lookup', args: { count: 42 } }] }), ...next.messages]);
  expect(graph.createRunStepResumeState().toolResultReleases).toHaveLength(2);
  const canonical = await validateToolMessageHistory(policy(), history, graph.createRunStepResumeState().toolResultReleases!);
  expect(canonical.filter((message) => message.getType() === 'tool')).toHaveLength(2); assertNoCanary(canonical);
});

it('never resurrects receipts for results absent from restored checkpoint history', async () => {
  class Graph extends StandardGraph { reduce(current: BaseMessage[], update: BaseMessage[]): BaseMessage[] { return this.reduceProtectedMessages(current, update); } }
  const fixture = await certifiedHistory(); const source = new Graph({ runId: 'stale-history', agents: [{ agentId: 'agent', provider: Providers.OPENAI }], toolResultProtection: fixture.gate });
  source.reduce([], fixture.messages); const state = source.createRunStepResumeState(); expect(state.toolResultReleases).toHaveLength(1);
  const restored = new Graph({ runId: 'stale-history', agents: [{ agentId: 'agent', provider: Providers.OPENAI }], toolResultProtection: policy() });
  restored.restoreRunStepResumeState(state);
  await restored.validateToolResultHistory([new HumanMessage('Allowed retained control')], state);
  expect(restored.createRunStepResumeState().toolResultReleases).toEqual([]);
});

it.each([false, true])('preserves validated supplied-history RunStart controls (selected=%s)', async (selected) => {
  const fixture = await certifiedHistory(); let observed = 0;
  const gate = selected ? fixture.gate : policy({ toolNames: ['other'] });
  const hooks = new HookRegistry(); hooks.register('RunStart', { hooks: [async (input) => { observed++; assertNoCanary(JSON.stringify(input)); return {}; }] });
  const run = await Run.create({ runId: `safe-supplied-${selected}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } }, toolResultProtection: gate, hooks });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] });
  const runStepState = run.Graph!.createRunStepResumeState(); runStepState.toolResultReleases = selected ? fixture.releases : [];
  await run.processStream({ messages: [...fixture.messages, new HumanMessage('Allowed control')], runStepState }, { version: 'v2', configurable: { thread_id: `safe-supplied-${selected}` } });
  expect(observed).toBe(1);
});

it.each(['iterator', 'proxy', 'subclass', 'inherited', 'renamed', 'control'] as const)('gates custom-checkpointer restored history before any model request (%s)', async (kind) => {
  let tamper = false; let reads = 0; let executions = 0; let models = 0;
  class Checkpointer extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(config);
      const messages = tuple?.checkpoint.channel_values.messages as BaseMessage[] | undefined;
      if (!tamper || tuple == null || messages == null || !messages.some((message) => message instanceof ToolMessage)) return tuple;
      let history = messages;
      if (kind === 'iterator') Object.defineProperty(history, Symbol.iterator, { value: function* () { reads++; if (reads > 1) yield* messages; } });
      if (kind === 'proxy') history = new Proxy(history, { get: (target, key, receiver): unknown => { reads++; return Reflect.get(target, key, receiver); } });
      if (kind === 'subclass') { class History extends Array<BaseMessage> {} history = new History(...messages); }
      if (kind === 'inherited') {
        const message = messages.find((entry) => entry instanceof ToolMessage)!; Reflect.deleteProperty(message, 'content');
        Object.setPrototypeOf(message, Object.create(ToolMessage.prototype, { content: { get: () => { reads++; throw new Error(fixtures.canaries[0]); } } }));
      }
      if (kind === 'renamed') for (const message of messages) {
        if (message instanceof AIMessage) for (const call of message.tool_calls ?? []) call.name = 'other';
        if (message instanceof ToolMessage) { message.name = 'other'; message.content = fixtures.canaries[0]; }
      }
      return { ...tuple, checkpoint: { ...tuple.checkpoint, channel_values: { ...tuple.checkpoint.channel_values, messages: history } } };
    }
  }
  const saver = new Checkpointer(); const runId = `restored-custom-${kind}`;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const create = (): Promise<Run<IState>> => Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'custom-source', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await first.processStream({ messages: [new HumanMessage('Allowed control')] }, config); expect(executions).toBe(1); tamper = true;
  const next = await create();
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { models++; assertNoCanary(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  next.Graph!.overrideModel = new Model({ responses: ['Allowed answer'] });
  const outcome = next.processStream({ messages: [new HumanMessage('Allowed next control')] }, config);
  if (kind === 'control') { await outcome; expect(models).toBe(1); }
  else { const error = await outcome.catch((value: Error) => value); expect(error).toMatchObject({ code: 'unsupported' }); assertNoCanary(String(error)); expect(models).toBe(0); }
  expect(reads).toBe(0); expect(executions).toBe(1);
});

it('keeps checkpoint receipt size bounded across compaction and rebuilt runs', async () => {
  const saver = new MemorySaver(); const runId = 'bounded-receipts'; let executions = 0;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const create = (): Promise<Run<IState>> => Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  for (let turn = 0; turn < 4; turn++) {
    const run = await create(); run.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: `bounded-${turn}`, name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
    await run.processStream({ messages: [createRemoveAllMessage(), new HumanMessage('Allowed compacted control')] }, config);
    const tuple = await saver.getTuple(config); const values = tuple!.checkpoint.channel_values as { messages: BaseMessage[]; runStepState: RunStepResumeState };
    expect(values.messages.filter((message) => message instanceof ToolMessage)).toHaveLength(1); expect(values.runStepState.toolResultReleases).toHaveLength(1); expect(values.runStepState.toolResultReleases![0].id).toBe(`bounded-${turn}`); assertNoCanary(values.messages);
  }
  expect(executions).toBe(4);
});

it('rejects a certified selected result disguised as another message role', async () => {
  class Graph extends StandardGraph { reduce(current: BaseMessage[], update: BaseMessage[]): BaseMessage[] { return this.reduceProtectedMessages(current, update); } }
  const fixture = await certifiedHistory(); const graph = new Graph({ runId: 'renamed-role', agents: [{ agentId: 'agent', provider: Providers.OPENAI }], toolResultProtection: fixture.gate });
  const messages = graph.reduce([], fixture.messages); const original = messages[1] as ToolMessage;
  const disguised = new HumanMessage({ id: original.id, content: fixtures.canaries[0] });
  await expect(validateToolMessageHistory(policy(), [messages[0], disguised], graph.createRunStepResumeState().toolResultReleases!)).rejects.toMatchObject({ code: 'unsupported' });
});

it.each((['warm', 'rebuilt'] as const).flatMap((runtime) => (['object', 'tuple', 'duplicate', 'overwrite', 'direct-command'] as const).map((shape) => [runtime, shape] as const)))('gates resume message updates before terminal graph publication (%s/%s)', async (runtime, shape) => {
  let executions = 0; let inspections = 0; const observed: string[] = [];
  const saver = new MemorySaver(); const runId = `resume-update-${runtime}-${shape}`;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const gate = policy({ inspect: ({ content }) => { inspections++; return approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } });
  const create = async (resumed: boolean): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval control' })] });
    const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: gate, skipCleanup: true });
    run.Graph!.overrideModel = new FakeChatModel({ responses: resumed ? ['Must not continue'] : [''], toolCalls: resumed ? [] : [{ id: 'pending-selected', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] }); return run;
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const first = await create(false); await first.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config); expect(first.getInterrupt()).toBeDefined(); expect(executions).toBe(0);
  const next = runtime === 'warm' ? first : await create(true);
  const forged = new ToolMessage({ id: 'forged-resume-result', name: 'lookup', tool_call_id: 'pending-selected', status: 'success', content: fixtures.canaries[0] });
  const saved = await saver.getTuple(config); const messages = saved!.checkpoint.channel_values.messages as BaseMessage[];
  let update: ConstructorParameters<typeof Command>[0]['update'] = { messages: [forged] };
  if (shape === 'tuple') update = [['messages', [forged]]];
  if (shape === 'duplicate') update = [['messages', [new HumanMessage('Allowed injected control')]], ['messages', [forged]]];
  if (shape === 'overwrite') update = { messages: new Overwrite([...messages, forged]) };
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (_chain, input): void => { observed.push(JSON.stringify(input)); }, handleChainEnd: (output): void => { observed.push(JSON.stringify(output)); } }); callback.awaitHandlers = true;
  const runConfig = { ...config, callbacks: [callback] };
  const pending = shape === 'direct-command' ? next.processStream(new Command({ resume: [{ type: 'approve' }], update }), runConfig) : next.resume([{ type: 'approve' }], runConfig, undefined, { update });
  const error = await pending.catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'incompatible' }); assertNoCanary(String(error)); expect(executions).toBe(0); expect(inspections).toBe(0); assertNoCanary(observed.join(''));
  const after = await saver.getTuple(config); assertNoCanary(after!.checkpoint.channel_values.messages);
});

it.each((['warm', 'rebuilt'] as const).flatMap((runtime) => (['none', 'human', 'tuple', 'duplicate', 'overwrite', 'unselected', 'certified'] as const).map((shape) => [runtime, shape] as const)))('preserves allowed terminal resume updates and execution authority (%s/%s)', async (runtime, shape) => {
  let executions = 0; const inspected: string[] = []; const saver = new MemorySaver(); const runId = `allowed-resume-${runtime}-${shape}`;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const gate = policy({ inspect: ({ content }) => { inspected.push(content); return approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } });
  const create = async (resumed: boolean): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
    const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: gate, skipCleanup: true });
    run.Graph!.overrideModel = new FakeChatModel({ responses: resumed ? ['Allowed final answer'] : [''], toolCalls: resumed ? [] : [{ id: 'allowed-pending', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] }); return run;
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const first = await create(false); await first.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config); expect(first.getInterrupt()).toBeDefined();
  const next = runtime === 'warm' ? first : await create(true);
  const injected = new HumanMessage('Allowed injected control');
  let update: ConstructorParameters<typeof Command>[0]['update'];
  if (shape === 'human') update = { messages: [injected] };
  if (shape === 'tuple') update = [['messages', [injected]]];
  if (shape === 'duplicate') update = [['messages', [injected]], ['messages', [new HumanMessage('Allowed second injection')]]];
  if (shape === 'overwrite') { const saved = await saver.getTuple(config); update = { messages: new Overwrite([...(saved!.checkpoint.channel_values.messages as BaseMessage[]), injected]) }; }
  if (shape === 'unselected') update = { messages: [new ToolMessage({ name: 'other', tool_call_id: 'other-control', content: 'Allowed unselected control' })] };
  if (shape === 'certified') { const node = new ToolNode({ tools: [lookup], toolResultProtection: gate }); const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'certified-source', name: 'lookup', args: { count: 42 } }] })] }) as { messages: ToolMessage[] }; update = { messages: result.messages }; }
  await next.resume([{ type: 'approve' }], config, undefined, { update });
  const checkpoint = await saver.getTuple(config); const messages = checkpoint!.checkpoint.channel_values.messages as BaseMessage[];
  expect(executions).toBe(shape === 'certified' ? 2 : 1); expect(inspected).toEqual(shape === 'certified' ? [fixtures.canaries[0], fixtures.canaries[0]] : [fixtures.canaries[0]]); assertNoCanary(messages); expect(JSON.stringify(messages)).toContain('[EMAIL_1]');
  if (!['none', 'unselected', 'certified'].includes(shape)) expect(JSON.stringify(messages)).toContain('Allowed injected control');
});

it.each(['array-proxy', 'messages-iterator', 'messages-accessor', 'update-accessor'] as const)('rejects controlled resume %s before graph callbacks', async (kind) => {
  let reads = 0; let executions = 0; const saver = new MemorySaver(); const runId = `hostile-update-${kind}`;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: policy(), skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'hostile-pending', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await run.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config);
  const messages = [new ToolMessage({ name: 'lookup', tool_call_id: 'hostile-pending', content: fixtures.canaries[0] })];
  let update: ConstructorParameters<typeof Command>[0]['update'] = { messages };
  if (kind === 'array-proxy') update = new Proxy<[string, unknown][]>([['messages', messages]], { get: (target, key, receiver): unknown => { reads++; return Reflect.get(target, key, receiver); } });
  if (kind === 'messages-iterator') Object.defineProperty(messages, Symbol.iterator, { value: function* () { reads++; yield* messages; } });
  if (kind === 'messages-accessor') Object.defineProperty(messages, '0', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
  if (kind === 'update-accessor') Object.defineProperty(update, 'messages', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
  const error = await run.processStream(new Command({ resume: [{ type: 'approve' }], update }), config).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0); expect(executions).toBe(0); assertNoCanary(String(error));
});

it.each((['warm', 'rebuilt'] as const).flatMap((runtime) => (['object', 'tuple', 'overwrite', 'direct'] as const).map((shape) => [runtime, shape] as const)))('admits alternate multi-agent resume messages before native callbacks (%s/%s)', async (runtime, shape) => {
  const saver = new MemorySaver(); const observed: string[] = []; let executions = 0; const runId = `alternate-resume-${runtime}-${shape}`;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const create = async (resumed: boolean): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
    const run = await Run.create({ runId, graphConfig: { type: 'multi-agent', agents: [{ agentId: 'worker', provider: Providers.OPENAI, tools: [lookup] }], edges: [], entryAgentId: 'worker', compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: policy(), skipCleanup: true });
    run.Graph!.overrideModel = new FakeChatModel({ responses: resumed ? ['Must not run'] : [''], toolCalls: resumed ? [] : [{ id: 'alternate-pending', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] }); return run;
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const first = await create(false); await first.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config); expect(first.getInterrupt()).toBeDefined();
  const next = runtime === 'warm' ? first : await create(true);
  const messages = [new ToolMessage({ id: 'uncertified-alternate', name: 'lookup', tool_call_id: 'alternate-pending', content: fixtures.canaries[0] })];
  let update: ConstructorParameters<typeof Command>[0]['update'] = { agentMessages: messages };
  if (shape === 'tuple') update = [['agentMessages', messages]];
  if (shape === 'overwrite') update = { agentMessages: new Overwrite(messages) };
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (_chain, input): void => { observed.push(JSON.stringify(input)); }, handleChainEnd: (output): void => { observed.push(JSON.stringify(output)); } }); callback.awaitHandlers = true;
  const options = { ...config, callbacks: [callback] };
  const pending = shape === 'direct' ? next.processStream(new Command({ resume: [{ type: 'approve' }], update }), options) : next.resume([{ type: 'approve' }], options, undefined, { update });
  const error = await pending.catch((value: Error) => value); expect(error).toMatchObject({ code: 'incompatible' }); expect(executions).toBe(0); assertNoCanary(String(error)); assertNoCanary(observed.join(''));
});

it.each((['warm', 'rebuilt'] as const).flatMap((runtime) => (['remove', 'remove-all', 'disguised'] as const).map((kind) => [runtime, kind] as const)))('distinguishes resume removal controls from retained protected messages (%s/%s)', async (runtime, kind) => {
  const saver = new MemorySaver(); const runId = `removal-resume-${runtime}-${kind}`; let executions = 0; let approvals = 0;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const approval = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed approval.', schema: z.object({ count: z.number() }), func: async () => { approvals++; return 'Approved control'; } });
  const create = async (): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'approval_control' ? 'ask' : 'allow', reason: 'Allowed approval' })] });
    return Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup, approval], toolEnd: true, compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: policy(), skipCleanup: true });
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'removable-source', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await first.processStream({ messages: [new HumanMessage('Allowed first control')] }, config);
  const committed = await saver.getTuple(config); const source = (committed!.checkpoint.channel_values.messages as BaseMessage[]).find((message) => message instanceof ToolMessage && message.tool_call_id === 'removable-source')!;
  first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'pending-approval', name: 'approval_control', args: { count: 42 }, type: 'tool_call' }] });
  await first.processStream({ messages: [new HumanMessage('Allowed next control')] }, config); expect(first.getInterrupt()).toBeDefined();
  const next = runtime === 'warm' ? first : await create();
  let messages: BaseMessage[] = [new HumanMessage({ id: source.id, content: fixtures.canaries[0] })];
  if (kind === 'remove') messages = [new RemoveMessage({ id: source.id! })];
  if (kind === 'remove-all') messages = [createRemoveAllMessage(), new HumanMessage('Allowed replacement history')];
  const pending = next.resume([{ type: 'approve' }], config, undefined, { update: { messages } });
  if (kind === 'disguised') { await expect(pending).rejects.toMatchObject({ code: 'unsupported' }); expect(approvals).toBe(0); }
  else { await pending; const after = await saver.getTuple(config); const retained = after!.checkpoint.channel_values.messages as BaseMessage[]; expect(approvals).toBe(1); assertNoCanary(retained); }
  expect(executions).toBe(1);
});

it.each(['individual', 'all'] as const)('validates identity-only removal controls before reduced history (%s)', async (kind) => {
  class Graph extends StandardGraph { reduce(current: BaseMessage[], update: BaseMessage[]): BaseMessage[] { return this.reduceProtectedMessages(current, update); } }
  const fixture = await certifiedHistory(); const graph = new Graph({ runId: `safe-removal-${kind}`, agents: [{ agentId: 'worker', provider: Providers.OPENAI }], toolResultProtection: fixture.gate });
  const messages = graph.reduce([], fixture.messages); const checkpoint = graph.createRunStepResumeState();
  const requested = kind === 'all' ? createRemoveAllMessage() : new RemoveMessage({ id: messages[1].id! });
  const control = snapshotToolMessageRemoval(requested)!;
  expect(control).toBeInstanceOf(RemoveMessage); expect(control).not.toBe(requested); expect(control.id).toBe(requested.id);
  const reduced = messagesStateReducer(messages, [control]);
  await expect(graph.validateToolResultHistory(reduced, checkpoint)).resolves.toEqual(reduced);
  expect(reduced.some((message) => message.id === messages[1].id)).toBe(false);
});

it.each(['content', 'metadata', 'accessor'] as const)('rejects payload-bearing removal %s without executing accessors', (kind) => {
  let reads = 0; const control = new RemoveMessage({ id: 'certified-id' });
  if (kind === 'content') control.content = fixtures.canaries[0];
  if (kind === 'metadata') control.additional_kwargs.output = fixtures.canaries[0];
  if (kind === 'accessor') Object.defineProperty(control, 'id', { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
  expect(() => snapshotToolMessageRemoval(control)).toThrow(ToolResultProtectionError); expect(reads).toBe(0);
});

it.each(['fresh', 'checkpoint', 'resume'] as const)('retains allowed alternate multi-agent message controls (%s)', async (path) => {
  const saver = new MemorySaver(); const runId = `alternate-control-${path}`; let executions = 0; const inputs: string[] = [];
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const run = await Run.create({ runId, graphConfig: { type: 'multi-agent', agents: [{ agentId: 'worker', provider: Providers.OPENAI, tools: [lookup] }], edges: [], entryAgentId: 'worker', compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: policy(), skipCleanup: true });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const alternate = [new HumanMessage('Allowed alternate control')];
  if (path === 'resume') {
    run.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'alternate-allowed', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
    await run.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config); expect(run.getInterrupt()).toBeDefined();
    run.Graph!.overrideModel = new Model({ responses: ['Allowed final answer'] });
    await run.resume([{ type: 'approve' }], config, undefined, { update: { agentMessages: alternate } }); expect(executions).toBe(1);
  } else {
    if (path === 'checkpoint') {
      run.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed checkpoint seed'] });
      await run.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config);
      const workflow = run.graphRunnable as typeof run.graphRunnable & { updateState(config: RunnableConfig, values: { agentMessages: BaseMessage[] }): Promise<unknown> };
      await workflow.updateState(config, { agentMessages: alternate });
    }
    run.Graph!.overrideModel = new Model({ responses: ['Allowed final answer'] });
    const state = path === 'fresh' ? { messages: [new HumanMessage('Allowed input')], agentMessages: alternate } : { messages: [new HumanMessage('Allowed next input')] };
    await run.processStream(state, config); expect(inputs.join('')).toContain('Allowed alternate control');
  }
  assertNoCanary(inputs.join('')); assertNoCanary(run.Graph!.getRunMessages());
});

it.each(['supplied', 'checkpoint'] as const)('validates alternate multi-agent history before startup observers (%s)', async (path) => {
  let hooks = 0; let models = 0; const saver = new MemorySaver(); const runId = `alternate-start-${path}`;
  class Checkpointer extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await saver.getTuple(config);
      if (tuple == null) return tuple;
      return { ...tuple, checkpoint: { ...tuple.checkpoint, channel_values: { ...tuple.checkpoint.channel_values, agentMessages: [new ToolMessage({ name: 'lookup', tool_call_id: 'raw-alternate', content: fixtures.canaries[0] })] } } };
    }
  }
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  if (path === 'checkpoint') {
    const seed = await Run.create({ runId, graphConfig: { type: 'multi-agent', agents: [{ agentId: 'worker', provider: Providers.OPENAI }], entryAgentId: 'worker', edges: [], compileOptions: { checkpointer: saver } } });
    seed.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed seed'] }); await seed.processStream({ messages: [new HumanMessage('Allowed input')] }, config);
  }
  const registry = new HookRegistry(); registry.register('RunStart', { hooks: [async () => { hooks++; return {}; }] });
  const run = await Run.create({ runId, graphConfig: { type: 'multi-agent', agents: [{ agentId: 'worker', provider: Providers.OPENAI }], entryAgentId: 'worker', edges: [], compileOptions: path === 'checkpoint' ? { checkpointer: new Checkpointer() } : undefined }, hooks: registry, toolResultProtection: policy() });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { models++; yield* super._streamResponseChunks(...args); } }
  run.Graph!.overrideModel = new Model({ responses: ['Must not run'] });
  const state = { messages: [new HumanMessage('Allowed input')], ...(path === 'supplied' ? { agentMessages: [new ToolMessage({ name: 'lookup', tool_call_id: 'raw-alternate', content: fixtures.canaries[0] })] } : {}) };
  const error = await run.processStream(state, config).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'incompatible' }); expect(hooks).toBe(0); expect(models).toBe(0); assertNoCanary(String(error));
});

it.each((['same-run', 'rebuilt'] as const).flatMap((runtime) => (['remove', 'all', 'disguised', 'payload'] as const).map((kind) => [runtime, kind] as const)))('admits fresh-turn removal controls separately from retained history (%s/%s)', async (runtime, kind) => {
  let executions = 0; let startup = 0; const inputs: string[] = []; const saver = new MemorySaver(); const runId = `fresh-removal-${runtime}-${kind}`;
  const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const create = async (): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('RunStart', { hooks: [async () => { startup++; return {}; }] });
    return Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, toolResultProtection: policy(), hooks });
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'fresh-removable', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await first.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config);
  const tuple = await saver.getTuple(config); const source = (tuple!.checkpoint.channel_values.messages as BaseMessage[]).find((message) => message instanceof ToolMessage && message.tool_call_id === 'fresh-removable')!;
  expect(source.id).toEqual(expect.any(String)); expect(executions).toBe(1); expect(startup).toBe(1);
  const next = runtime === 'same-run' ? first : await create();
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  next.Graph!.overrideModel = new Model({ responses: ['Allowed final answer'] });
  let update: BaseMessage[] = [new HumanMessage({ id: source.id, content: fixtures.canaries[0] })];
  if (kind === 'remove') update = [new RemoveMessage({ id: source.id! }), new HumanMessage('Allowed deletion control')];
  if (kind === 'all') update = [createRemoveAllMessage(), new HumanMessage('Allowed compacted control')];
  if (kind === 'payload') { const removal = new RemoveMessage({ id: source.id! }); removal.content = fixtures.canaries[0]; update = [removal]; }
  const result = next.processStream({ messages: update }, config);
  if (kind === 'disguised' || kind === 'payload') { const error = await result.catch((value: Error) => value); expect(error).toMatchObject({ code: 'unsupported' }); assertNoCanary(String(error)); expect(startup).toBe(1); expect(inputs).toEqual([]); }
  else {
    await result; expect(startup).toBe(2); expect(inputs).toHaveLength(1); assertNoCanary(inputs[0]); expect(inputs[0]).not.toContain('[EMAIL_1]');
    const retained = (await saver.getTuple(config))!.checkpoint.channel_values as { messages: BaseMessage[]; runStepState: RunStepResumeState };
    expect(retained.messages.some((message) => message.id === source.id)).toBe(false); expect(retained.runStepState.toolResultReleases).toEqual([]); assertNoCanary(retained.messages);
  }
  expect(executions).toBe(1);
});

it.each(['renamed', 'additional_kwargs', 'response_metadata', 'own-accessor', 'control'] as const)('admits only own currently selected checkpoint receipts (%s)', async (kind) => {
  let tamper = false; let reads = 0; let executions = 0; let hooks = 0; let models = 0; const runId = `receipt-admission-${kind}`;
  class Checkpointer extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(config);
      if (!tamper || tuple == null) return tuple;
      const values = tuple.checkpoint.channel_values as { messages?: BaseMessage[]; runStepState?: RunStepResumeState };
      const release = values.runStepState?.toolResultReleases?.[0];
      if (release == null) return tuple;
      if (kind === 'renamed') {
        Object.defineProperty(release, 'name', { value: 'other', enumerable: true }); release.envelope.name = 'other';
        for (const message of values.messages ?? []) {
          if (message instanceof AIMessage) for (const call of message.tool_calls ?? []) call.name = 'other';
          if (message instanceof ToolMessage) { message.name = 'other'; message.content = fixtures.canaries[0]; }
        }
      } else if (kind !== 'control') {
        const field = kind === 'own-accessor' ? 'additional_kwargs' : kind;
        const getter = (): never => { reads++; throw new Error(fixtures.canaries[0]); };
        if (kind === 'own-accessor') Object.defineProperty(release.envelope, field, { get: getter });
        else { Reflect.deleteProperty(release.envelope, field); Object.setPrototypeOf(release.envelope, { get [field]() { return getter(); } }); }
      }
      return tuple;
    }
  }
  const saver = new Checkpointer(); const lookup = direct(() => { executions++; return fixtures.canaries[0]; });
  const create = (): Promise<Run<IState>> => {
    const registry = new HookRegistry(); registry.register('RunStart', { hooks: [async () => { hooks++; return {}; }] });
    return Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], toolEnd: true, compileOptions: { checkpointer: saver } }, toolResultProtection: policy(), hooks: registry });
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'receipt-source', name: 'lookup', args: { count: 42 }, type: 'tool_call' }] });
  await first.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config); expect(executions).toBe(1); tamper = true;
  const next = await create();
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { models++; assertNoCanary(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  next.Graph!.overrideModel = new Model({ responses: ['Allowed final answer'] });
  const result = next.processStream({ messages: [new HumanMessage('Allowed next control')] }, config);
  if (kind === 'control') { await result; expect(hooks).toBe(2); expect(models).toBe(1); }
  else { const error = await result.catch((value: Error) => value); expect(error).toMatchObject({ code: kind === 'own-accessor' ? 'unsupported' : 'incompatible' }); expect(hooks).toBe(1); expect(models).toBe(0); assertNoCanary(String(error)); }
  expect(reads).toBe(0); expect(executions).toBe(1);
});

it('rejects historical receipts outside a narrowed protected set before inspection', async () => {
  const fixture = await certifiedHistory(); let inspections = 0;
  await expect(validateToolMessageHistory(policy({ toolNames: ['other'], inspect: () => { inspections++; return approve('Must not inspect'); } }), fixture.messages, fixture.releases)).rejects.toMatchObject({ code: 'incompatible' });
  expect(inspections).toBe(0);
});

it.each((['call-renamed', 'receipt-renamed', 'all-renamed', 'id-renamed', 'future', 'command-output', 'selected', 'unselected'] as const).flatMap((kind) => [false, true].map((references) => [kind, references] as const)))('admits embedded replay receipts before unselected classification (%s/refs=%s)', async (kind, references) => {
  let sourceExecutions = 0; let otherExecutions = 0; let approvals = 0; const inspections: string[] = []; const events: string[] = [];
  const gate = policy({ inspect: ({ content }) => { inspections.push(content); return approve(content.replaceAll(fixtures.canaries[0], '[EMAIL_1]')); } });
  const lookup = direct(() => { sourceExecutions++; return fixtures.canaries[0]; });
  const other = new DynamicStructuredTool({ name: 'other', description: 'Allowed unselected tool.', schema: z.object({ count: z.number() }), func: async () => { otherExecutions++; return 'Allowed unselected control'; } });
  const approval = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed approval control.', schema: z.object({ count: z.number() }), func: async () => { approvals++; return 'Approved control'; } });
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'approval_control' ? 'ask' : 'allow', reason: 'Allowed approval' })] });
  const saver = new MemorySaver(); const runId = `embedded-release-${kind}-${references}`;
  const compile = (): Pick<Runnable<unknown, unknown>, 'invoke'> => new StateGraph(MessagesAnnotation).addNode('tools', new ToolNode({ trace: true, tools: [lookup, other, approval], hookRegistry: hooks, humanInTheLoop: { enabled: true }, toolResultProtection: gate, toolOutputReferences: { enabled: references } })).addEdge(START, 'tools').addEdge('tools', END).compile({ checkpointer: saver });
  const calls: ToolCall[] = [{ id: 'settled-control', name: kind === 'unselected' ? 'other' : 'lookup', args: { count: 42 }, type: 'tool_call' }, { id: 'approval-control', name: 'approval_control', args: { count: 42 }, type: 'tool_call' }];
  const config = { configurable: { thread_id: runId, run_id: runId }, callbacks: [observer(events)] };
  const paused = await compile().invoke({ messages: [new AIMessage({ id: 'embedded-batch', content: '', tool_calls: calls })] }, config);
  if (!isInterrupted(paused)) throw new Error('Expected approval checkpoint');
  const replay = getToolBatchReplayState(paused.__interrupt__[0].value); if (replay == null) throw new Error('Expected settled replay');
  const record = replay.records[0]; const results = await saver.serde.loadsTyped(record.encoding, Buffer.from(record.data, 'base64')) as Array<[string, SettledToolBatchResult]>;
  const result = results[0][1]; expect(result.release != null).toBe(kind !== 'unselected');
  if (kind !== 'selected' && kind !== 'unselected') {
    if (!(result.output instanceof ToolMessage)) throw new Error('Expected settled tool output');
    result.output.content = fixtures.canaries[0];
    if (kind === 'call-renamed' || kind === 'all-renamed') { calls[0] = { ...calls[0], name: 'other' }; result.proposal.name = 'other'; result.output.name = 'other'; }
    if (kind === 'receipt-renamed' || kind === 'all-renamed') { Object.defineProperty(result.release!, 'name', { value: 'other', enumerable: true }); result.release!.envelope.name = 'other'; }
    if (kind === 'id-renamed') { calls[0] = { ...calls[0], id: 'renamed-control' }; results[0][0] = JSON.stringify(['id', 'renamed-control']); result.output.tool_call_id = 'renamed-control'; }
    if (kind === 'future') Object.defineProperty(result.release!, 'version', { value: 2 });
    if (kind === 'command-output') result.output = new Command({ update: { messages: [result.output] } });
  }
  const [encoding, bytes] = await saver.serde.dumpsTyped(results); record.encoding = encoding; record.data = Buffer.from(bytes).toString('base64');
  const [scope, id] = JSON.parse(record.batch) as [string, string, string];
  record.batch = JSON.stringify([scope, id, createHash('sha256').update(stableStringify(calls)).digest('hex')]);
  events.length = 0;
  const pending = compile().invoke(new Command({ resume: [{ type: 'approve' }], update: { messages: [new AIMessage({ id: 'embedded-batch', content: '', tool_calls: calls })] } }), { ...config, configurable: { ...config.configurable, [TOOL_BATCH_REPLAY_KEY]: replay } });
  if (kind === 'selected' || kind === 'unselected') {
    const resumed = await pending; assertNoCanary(resumed); expect(JSON.stringify(resumed)).toContain(kind === 'selected' ? '[EMAIL_1]' : 'Allowed unselected control'); expect(approvals).toBe(1);
  } else { const error = await pending.catch((value: Error) => value); expect(error).toMatchObject({ code: ['all-renamed', 'receipt-renamed', 'future'].includes(kind) ? 'incompatible' : 'unsupported' }); assertNoCanary(String(error)); expect(approvals).toBe(0); }
  expect(sourceExecutions).toBe(kind === 'unselected' ? 0 : 1); expect(otherExecutions).toBe(kind === 'unselected' ? 1 : 0); assertNoCanary(events.join(''));
  let expected = [fixtures.canaries[0]];
  if (kind === 'selected') expected = [fixtures.canaries[0], '[EMAIL_1]'];
  if (kind === 'unselected') expected = [];
  expect(inspections).toEqual(expected);
});

it.each(['values', 'messages', 'agentMessages', 'tasks', 'task-state', 'interrupts', 'release-list'] as const)('rejects checkpoint container %s accessors before any getter executes', async (field) => {
  let reads = 0;
  const graph = new StandardGraph({ runId: `container-${field}`, agents: [{ agentId: 'worker', provider: Providers.OPENAI }], toolResultProtection: policy() });
  const snapshot = { values: { messages: [new HumanMessage('Allowed control')], runStepState: graph.createRunStepResumeState() }, tasks: [{ state: { values: { messages: [new HumanMessage('Allowed nested control')] }, tasks: [] }, interrupts: [] }] };
  let container: object = snapshot; let key: string = field;
  if (field === 'messages' || field === 'agentMessages') container = snapshot.values;
  if (field === 'task-state') { container = snapshot.tasks[0]; key = 'state'; }
  if (field === 'interrupts') container = snapshot.tasks[0];
  if (field === 'release-list') { container = snapshot.values.runStepState; key = 'toolResultReleases'; }
  Object.defineProperty(container, key, { get: () => { reads++; throw new Error(fixtures.canaries[0]); } });
  const error = await graph.validateToolResultCheckpoint(snapshot).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0); assertNoCanary(String(error));
});

it.each(['own', 'inherited', 'stateful', 'nested'] as const)('rejects custom workflow checkpoint containers before startup/chain exposure (%s)', async (kind) => {
  let reads = 0; let startup = 0; let models = 0; const saver = new MemorySaver(); const runId = `checkpoint-container-${kind}`;
  const registry = new HookRegistry(); registry.register('RunStart', { hooks: [async () => { startup++; return {}; }] });
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } }, toolResultProtection: policy(), hooks: registry });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { models++; yield* super._streamResponseChunks(...args); } }
  run.Graph!.overrideModel = new Model({ responses: ['Must not run'] });
  const workflow = run.graphRunnable as typeof run.graphRunnable & { getState(config: RunnableConfig): Promise<{ values: { messages: BaseMessage[] }; tasks: [] }> };
  const values = { messages: [new HumanMessage('Allowed checkpoint input')] };
  const getter = (): BaseMessage[] => { reads++; if (kind === 'stateful' && reads === 1) return values.messages; throw new Error(fixtures.canaries[0]); };
  let snapshot: object = { values, tasks: [] };
  if (kind === 'inherited') snapshot = { values: Object.create({ get messages() { return getter(); } }), tasks: [] };
  else if (kind === 'nested') snapshot = { values, tasks: [{ state: { values: Object.defineProperty({}, 'messages', { get: getter }), tasks: [] } }] };
  else Object.defineProperty(values, 'messages', { get: getter });
  workflow.getState = async () => snapshot as Awaited<ReturnType<typeof workflow.getState>>;
  const error = await run.processStream({ messages: [new HumanMessage('Allowed supplied input')] }, { version: 'v2', configurable: { thread_id: runId } }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0); expect(startup).toBe(0); expect(models).toBe(0); assertNoCanary(String(error));
});

it('uses the public tool-result error class for a misplaced required tool policy', async () => {
  const config = { runId: 'misplaced-tool-policy', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, toolResultProtection: policy() } } as unknown as Parameters<typeof Run.create>[0];
  const error = await Run.create(config).catch((value: Error) => value);
  expect(error).toBeInstanceOf(ToolResultProtectionError); expect(error).toMatchObject({ code: 'incompatible' }); assertNoCanary(String(error));
});

it('returns detached validated checkpoint containers without mutating a frozen custom snapshot', async () => {
  const gate = policy(); const graph = new StandardGraph({ runId: 'frozen-checkpoint', agents: [{ agentId: 'worker', provider: Providers.OPENAI }], toolResultProtection: gate });
  const messages = [new HumanMessage('Allowed frozen control')]; Object.freeze(messages);
  const nestedMessages = [new HumanMessage('Allowed nested frozen control')]; Object.freeze(nestedMessages);
  const values = Object.freeze({ messages, runStepState: Object.freeze(graph.createRunStepResumeState()) });
  const state = Object.freeze({ values: Object.freeze({ messages: nestedMessages }), tasks: Object.freeze([]) });
  const task = Object.freeze({ state, interrupts: Object.freeze([{ id: 'allowed-interrupt', value: 'Allowed approval control' }]) });
  const input = Object.freeze({ values, tasks: Object.freeze([task]) });
  const snapshot = await graph.validateToolResultCheckpoint(input);
  expect(snapshot).not.toBe(input); expect(snapshot.values).not.toBe(values); expect(snapshot.tasks).not.toBe(input.tasks); expect(snapshot.tasks[0]).not.toBe(task); expect(snapshot.tasks[0].state).not.toBe(state);
  expect(snapshot.values.messages).toEqual(values.messages); expect(snapshot.tasks[0].interrupts).toEqual(task.interrupts);
});

it('rejects cyclic checkpoint task containers with a raw-free protection error', async () => {
  const graph = new StandardGraph({ runId: 'cyclic-checkpoint', agents: [{ agentId: 'worker', provider: Providers.OPENAI }], toolResultProtection: policy() });
  type Snapshot = { values: { messages: BaseMessage[] }; tasks: Array<{ state: Snapshot }> };
  const input: Snapshot = { values: { messages: [new HumanMessage('Allowed control')] }, tasks: [] }; input.tasks.push({ state: input });
  await expect(graph.validateToolResultCheckpoint(input)).rejects.toMatchObject({ code: 'unsupported' });
});

it.each(['empty', 'populated', 'explicit', 'mapped', 'unprotected'] as const)('executes only the checkpoint admitted before startup awaits (%s)', async (mode) => {
  const saver = new MemorySaver(); const runId = `checkpoint-race-${mode}`; const observed: string[] = []; const inputs: string[] = [];
  const base = { version: 'v2' as const, configurable: { thread_id: runId, user_id: 'allowed-principal', resource: 'allowed-resource' } };
  if (mode !== 'empty') {
    const seed = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } } });
    seed.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed initial answer'] });
    await seed.processStream({ messages: [new HumanMessage('Allowed initial input')] }, base);
  }
  const admitted = await saver.getTuple(base);
  let writerCheckpoint = '';
  const hooks = new HookRegistry(); hooks.register('RunStart', { hooks: [async () => {
    const writer = await Run.create({ runId: `legacy-writer-${mode}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } } });
    writer.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed writer answer'] });
    await writer.processStream({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'legacy-raced-source', name: 'lookup', args: { count: 42 } }] }), new ToolMessage({ id: 'legacy-raced-result', name: 'lookup', tool_call_id: 'legacy-raced-source', content: fixtures.canaries[0] }), new HumanMessage('Allowed writer input')] }, base);
    writerCheckpoint = (await saver.getTuple(base))!.checkpoint.id;
    expect(writerCheckpoint).not.toBe(admitted?.checkpoint.id);
    return {};
  }] });
  class Model extends FakeChatModel {
    override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); }
  }
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } }, toolResultProtection: mode === 'unprotected' ? undefined : policy(), hooks });
  run.Graph!.overrideModel = new Model({ responses: ['Allowed reader answer'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (_chain, input): void => { observed.push(JSON.stringify(input)); }, handleChainEnd: (output): void => { observed.push(JSON.stringify(output)); } }); callback.awaitHandlers = true;
  const caller = { ...base, configurable: { ...base.configurable, ...(mode === 'explicit' ? { checkpoint_id: admitted!.checkpoint.id } : {}), ...(mode === 'mapped' ? { checkpoint_map: { '': admitted!.checkpoint.id } } : {}) }, callbacks: [callback] };
  const outcome = await run.processStream({ messages: [new HumanMessage('Allowed reader input')] }, caller).catch((error: Error) => error);
  if (mode !== 'unprotected') assertNoCanary(observed.join(''));
  expect(outcome).toBeUndefined(); expect(inputs).toHaveLength(1);
  if (mode === 'unprotected') expect(inputs[0]).toContain(fixtures.canaries[0]);
  else { assertNoCanary(inputs[0]); assertNoCanary(observed.join('')); }
  const result = (await saver.getTuple(base))!; const history = result.checkpoint.channel_values.messages as BaseMessage[];
  if (mode !== 'unprotected') { assertNoCanary(history); expect(result.parentConfig?.configurable?.checkpoint_id).not.toBe(writerCheckpoint); }
  expect(caller.configurable).toMatchObject({ thread_id: runId, user_id: 'allowed-principal', resource: 'allowed-resource' });
  if (mode === 'explicit') expect(caller.configurable).toHaveProperty('checkpoint_id', admitted!.checkpoint.id);
  else expect(caller.configurable).not.toHaveProperty('checkpoint_id');
  if (mode === 'mapped') expect(caller.configurable).toHaveProperty('checkpoint_map', { '': admitted!.checkpoint.id });
});

it.each(['missing', 'foreign-thread', 'namespace'] as const)('fails closed before startup when an admitted checkpoint cannot bind execution (%s)', async (kind) => {
  let hooks = 0; let models = 0;
  const registry = new HookRegistry(); registry.register('RunStart', { hooks: [async () => { hooks++; return {}; }] });
  const runId = `unbound-checkpoint-${kind}`;
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: new MemorySaver() } }, toolResultProtection: policy(), hooks: registry });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { models++; yield* super._streamResponseChunks(...args); } }
  run.Graph!.overrideModel = new Model({ responses: ['Must not run'] });
  const workflow = run.graphRunnable as typeof run.graphRunnable & { getState(config: RunnableConfig): Promise<{ values: { messages: BaseMessage[] }; config: RunnableConfig; tasks: [] }> };
  workflow.getState = async () => ({ values: { messages: [new HumanMessage('Allowed checkpoint control')] }, config: { configurable: { thread_id: kind === 'foreign-thread' ? 'foreign-thread' : runId, ...(kind === 'missing' ? {} : { checkpoint_id: 'allowed-checkpoint' }), checkpoint_ns: kind === 'namespace' ? 'child:control' : '' } }, tasks: [] });
  const error = await run.processStream({ messages: [new HumanMessage('Allowed supplied control')] }, { version: 'v2', configurable: { thread_id: runId } }).catch((value: Error) => value);
  expect(error).toMatchObject({ code: kind === 'missing' ? 'incompatible' : 'unsupported' }); expect(hooks).toBe(0); expect(models).toBe(0); assertNoCanary(String(error));
});

it('pins foreground child startup to its admitted empty checkpoint despite a competing writer', async () => {
  const thread = 'child-checkpoint-race'; let childConfig: RunnableConfig | undefined; let startups = 0; const childInputs: string[] = [];
  class Saver extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      if (config.configurable?.thread_id !== thread) childConfig = { configurable: { thread_id: config.configurable?.thread_id } };
      return await super.getTuple(config);
    }
  }
  const saver = new Saver(); const registry = new HookRegistry();
  registry.register('SubagentStart', { hooks: [async () => {
    startups++; if (childConfig == null || typeof childConfig.configurable?.thread_id !== 'string') throw new Error('Expected child checkpoint scope');
    const writer = await Run.create({ runId: 'legacy-child-writer', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } } });
    writer.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed writer answer'] });
    await writer.processStream({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'legacy-child-source', name: 'lookup', args: { count: 42 } }] }), new ToolMessage({ name: 'lookup', tool_call_id: 'legacy-child-source', content: fixtures.canaries[0] }), new HumanMessage('Allowed writer input')] }, { version: 'v2', ...childConfig });
    return {};
  }] });
  class ChildModel extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { childInputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  const run = await Run.create({ runId: thread, graphConfig: { type: 'standard', agents: [{ agentId: 'parent', provider: Providers.OPENAI, instructions: 'Allowed parent control.', subagentConfigs: [{ type: 'worker', name: 'Worker', description: 'Allowed child control.', agentInputs: { agentId: 'child', provider: Providers.OPENAI, instructions: 'Allowed child instructions.' } }] }], compileOptions: { checkpointer: saver } }, hooks: registry, humanInTheLoop: { enabled: true }, toolResultProtection: policy(), skipCleanup: true });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed final parent answer'], toolCalls: [{ id: 'child-race-control', name: Constants.SUBAGENT, args: { description: 'Allowed child task.', subagent_type: 'worker' }, type: 'tool_call' }] });
  run.Graph!.setSubagentModelOverride(new ChildModel({ responses: ['Allowed child answer'] }));
  await run.processStream({ messages: [new HumanMessage('Allowed parent input')] }, { version: 'v2', configurable: { thread_id: thread } });
  expect(startups).toBe(1); expect(childInputs).toHaveLength(1); assertNoCanary(childInputs[0]); assertNoCanary(run.Graph!.getRunMessages());
});

it.each((['warm', 'rebuilt'] as const).flatMap((runtime) => (['mismatch', 'concurrent', 'matching', 'unknown', 'unprotected'] as const).map((mode) => [runtime, mode] as const)))('binds the exact admitted descendant before interrupted graph resume (%s/%s)', async (runtime, mode) => {
  let approvals = 0; const inputs: string[] = []; const observed: string[] = []; const saver = new MemorySaver(); const runId = `descendant-cursor-${runtime}-${mode}`;
  const approval = new DynamicStructuredTool({ name: 'approval_control', description: 'Allowed approval control.', schema: z.object({ count: z.number() }), func: async () => { approvals++; return 'Approved control'; } });
  const create = async (resumed: boolean): Promise<Run<IState>> => {
    const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
    const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [approval], compileOptions: { checkpointer: saver } }, hooks, humanInTheLoop: { enabled: true }, toolResultProtection: mode === 'unprotected' ? undefined : policy(), skipCleanup: true });
    run.Graph!.overrideModel = new FakeChatModel({ responses: resumed ? ['Allowed final answer'] : ['', 'Allowed final answer'], toolCalls: resumed ? [] : [{ id: 'descendant-pending', name: 'approval_control', args: { count: 42 }, type: 'tool_call' }] });
    return run;
  };
  const config = { version: 'v2' as const, configurable: { thread_id: runId, user_id: 'allowed-principal' } };
  const first = await create(false); await first.processStream({ messages: [new HumanMessage('Allowed initial control')] }, config); expect(first.getInterrupt()).toBeDefined();
  type Snapshot = { config: RunnableConfig; values: { messages: BaseMessage[] }; tasks: Array<{ state?: Snapshot }> };
  const workflow = first.graphRunnable as typeof first.graphRunnable & { getState(config: RunnableConfig, options: { subgraphs: boolean }): Promise<Snapshot> };
  const admitted = await workflow.getState(config, { subgraphs: true }); const child = admitted.tasks.find((task) => task.state?.config.configurable?.checkpoint_ns != null)?.state;
  if (child == null) throw new Error('Expected interrupted agent descendant');
  const childConfig = child.config; const namespace = childConfig.configurable!.checkpoint_ns as string; const safeId = childConfig.configurable!.checkpoint_id as string;
  const tuple = await saver.getTuple(childConfig); if (tuple == null || tuple.metadata == null) throw new Error('Expected descendant checkpoint');
  const legacyId = '00000000-0000-1000-8000-000000000001'; const lateId = 'ffffffff-ffff-1fff-8fff-ffffffffffff';
  const writeRaw = async (id: string): Promise<void> => {
    const checkpoint = { ...tuple.checkpoint, id, channel_values: { ...tuple.checkpoint.channel_values, messages: [...(tuple.checkpoint.channel_values.messages as BaseMessage[]), new AIMessage({ content: '', tool_calls: [{ id: 'legacy-descendant-source', name: 'lookup', args: { count: 42 } }] }), new ToolMessage({ id: 'legacy-descendant-result', name: 'lookup', tool_call_id: 'legacy-descendant-source', content: fixtures.canaries[0] })] } };
    await saver.put({ configurable: { thread_id: runId, checkpoint_ns: namespace } }, checkpoint, tuple.metadata!);
  };
  await writeRaw(legacyId);
  expect((await saver.getTuple({ configurable: { thread_id: runId, checkpoint_ns: namespace } }))!.checkpoint.id).toBe(safeId);
  const next = runtime === 'warm' ? first : await create(true);
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { inputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  next.Graph!.overrideModel = new Model({ responses: ['Allowed final answer'] });
  let wrote = false;
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: async (_chain, input): Promise<void> => {
    if (mode === 'concurrent' && !wrote) { wrote = true; await writeRaw(lateId); }
    observed.push(JSON.stringify(input));
  }, handleChainEnd: (output): void => { observed.push(JSON.stringify(output)); } }); callback.awaitHandlers = true;
  let checkpointMap: Record<string, string> = {};
  if (mode === 'mismatch' || mode === 'unprotected') checkpointMap = { [namespace]: legacyId };
  if (mode === 'matching') checkpointMap = { [namespace]: safeId };
  if (mode === 'unknown') checkpointMap = { 'unknown:descendant': legacyId };
  const caller = { ...config, configurable: { ...config.configurable, checkpoint_map: checkpointMap }, callbacks: [callback] };
  const outcome = await next.resume([{ type: 'approve' }], caller).catch((value: Error) => value);
  if (mode === 'mismatch' || mode === 'unknown') { expect(outcome).toMatchObject({ code: 'unsupported' }); expect(approvals).toBe(0); expect(inputs).toEqual([]); expect(observed).toEqual([]); }
  else if (mode === 'unprotected') {
    expect(outcome).toBeUndefined(); expect(approvals).toBe(0); expect(inputs).toHaveLength(1); expect(inputs[0]).toContain(fixtures.canaries[0]);
  } else {
    expect(outcome).toBeUndefined(); expect(approvals).toBe(1); expect(inputs).toHaveLength(1);
    assertNoCanary(inputs[0]); assertNoCanary(observed.join(''));
    if (mode === 'concurrent') expect(wrote).toBe(true);
  }
  expect(caller.configurable.checkpoint_map).toEqual(checkpointMap); expect(caller.configurable.user_id).toBe('allowed-principal');
});

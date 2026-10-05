import { z } from 'zod';
import { createServer } from 'node:http';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  GraphInterrupt,
  Command,
  StateGraph,
  MessagesAnnotation,
  MemorySaver,
  START,
  END,
  isInterrupted,
} from '@langchain/langgraph';
import type { AddressInfo } from 'node:net';
import type {
  ToolResultProtection,
  ToolResultProtectionResult,
} from '@/protection/toolResult';
import type { ToolExecuteBatchRequest, EventHandler } from '@/types';
import { createLocalProgrammaticToolCallingTool } from '@/tools/local/LocalProgrammaticToolCalling';
import {
  createProgrammaticToolCallingTool,
  executeTools,
} from '@/tools/ProgrammaticToolCalling';
import { createBashProgrammaticToolCallingTool } from '@/tools/BashProgrammaticToolCalling';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import fixtures from '@/protection/__tests__/fixtures/a1.json';
import { Constants, GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
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

import { createServer } from 'node:http';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import type { AddressInfo } from 'node:net';
import type {
  HostToolDescriptor,
  HostToolRequest,
  HostToolService,
} from './index';
import type { ToolExecuteBatchRequest, ToolExecuteResult } from '@/types';
import { GraphEvents, Providers } from '@/common';
import { createHostToolBindings } from './index';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
  tenantId: 'tenant',
};
const tools: HostToolDescriptor[] = [
  {
    reference: 'authorized:echo',
    definition: {
      name: 'echo',
      description: 'Echo text',
      parameters: {
        type: 'object' as const,
        properties: { text: { type: 'string' } },
        required: ['text'],
      },
    },
  },
];
const caller = {
  version: 'v2' as const,
  configurable: { thread_id: 'thread' },
};

function dispatch(
  service: HostToolService,
  calls: ToolExecuteBatchRequest['toolCalls'],
  signal?: AbortSignal
): Promise<ToolExecuteResult[]> {
  const bindings = createHostToolBindings({ identity, tools, service });
  return new Promise((resolve, reject) => {
    void bindings.customHandlers[GraphEvents.ON_TOOL_EXECUTE].handle(
      GraphEvents.ON_TOOL_EXECUTE,
      { toolCalls: calls, signal, resolve, reject }
    );
  });
}

describe('host tool harness', () => {
  it('keeps host authority separate from model arguments and canonicalizes result order', async () => {
    const requests: HostToolRequest[] = [];
    const service: HostToolService = {
      version: 1,
      invoke: async (request) => {
        requests.push(request);
        return request.calls
          .map(({ call }) => ({
            toolCallId: call.id,
            status: 'success' as const,
            content: 'ok',
          }))
          .reverse();
      },
    };
    const result = await dispatch(service, [
      {
        id: 'one',
        name: 'echo',
        args: { reference: 'unauthorized', userId: 'other' },
      },
      { id: 'two', name: 'echo', args: {} },
    ]);
    expect(result.map((r) => r.toolCallId)).toEqual(['one', 'two']);
    expect(requests[0].identity).toEqual(identity);
    expect(requests[0].calls[0].reference).toBe('authorized:echo');
    expect(requests[0]).not.toHaveProperty('configurable');
  });
  it.each([
    { calls: [{ id: 'one', name: 'unknown', args: {} }] },
    {
      calls: [
        { id: 'one', name: 'echo', args: {} },
        { id: 'one', name: 'echo', args: {} },
      ],
    },
  ])(
    'rejects an unauthorized batch before invoking the service',
    async ({ calls }) => {
      const invoke = jest.fn();
      await expect(dispatch({ version: 1, invoke }, calls)).rejects.toThrow(
        'Host tool execution failed'
      );
      expect(invoke).not.toHaveBeenCalled();
    }
  );
  it.each([
    { results: [] },
    {
      results: [
        { toolCallId: 'other', status: 'success' as const, content: 'bad' },
      ],
    },
  ])('rejects missing or foreign result receipts', async ({ results }) => {
    await expect(
      dispatch({ version: 1, invoke: async () => results }, [
        { id: 'one', name: 'echo', args: {} },
      ])
    ).rejects.toThrow('Host tool execution failed');
  });
  it('does not retry uncertain failures or disclose the service error', async () => {
    const invoke = jest.fn(async () => {
      throw new Error('sensitive upstream credential');
    });
    await expect(
      dispatch({ version: 1, invoke }, [{ id: 'one', name: 'echo', args: {} }])
    ).rejects.toThrow(/^Host tool execution failed$/);
    expect(invoke).toHaveBeenCalledTimes(1);
  });
  it('does not dispatch cancelled work and forwards live cancellation', async () => {
    const controller = new AbortController();
    controller.abort();
    const invoke = jest.fn();
    await expect(
      dispatch(
        { version: 1, invoke },
        [{ id: 'one', name: 'echo', args: {} }],
        controller.signal
      )
    ).rejects.toThrow();
    expect(invoke).not.toHaveBeenCalled();
    const active = new AbortController();
    await dispatch(
      {
        version: 1,
        invoke: async (_request, signal) => {
          expect(signal).toBe(active.signal);
          return [{ toolCallId: 'one', status: 'success', content: 'ok' }];
        },
      },
      [{ id: 'one', name: 'echo', args: {} }],
      active.signal
    );
  });
  it('projects only public execution identity fields from a richer host object', async () => {
    const hostIdentity = { ...identity, credential: 'private' };
    const service: HostToolService = {
      version: 1,
      invoke: async (request) => {
        expect(request.identity).not.toHaveProperty('credential');
        return [
          {
            toolCallId: 'one',
            status: 'success',
            content: 'ok',
            received_at: 1,
          },
        ];
      },
    };
    const bindings = createHostToolBindings({
      identity: hostIdentity,
      tools,
      service,
    });
    const result = await new Promise<ToolExecuteResult[]>((resolve, reject) => {
      void bindings.customHandlers[GraphEvents.ON_TOOL_EXECUTE].handle(
        GraphEvents.ON_TOOL_EXECUTE,
        { toolCalls: [{ id: 'one', name: 'echo', args: {} }], resolve, reject }
      );
    });
    expect(result[0]).not.toHaveProperty('received_at');
  });
  it('rejects duplicate descriptors and isolates bindings from subsequent mutation', () => {
    const service: HostToolService = { version: 1, invoke: async () => [] };
    expect(() =>
      createHostToolBindings({ identity, tools: [...tools, ...tools], service })
    ).toThrow('duplicate');
    const bindings = createHostToolBindings({ identity, tools, service });
    bindings.toolDefinitions[0].name = 'changed';
    expect(tools[0].definition.name).toBe('echo');
  });

  it('forwards executing agent and publishes fast results before batch settlement', async () => {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let first!: () => void;
    const ready = new Promise<void>((resolve) => {
      first = resolve;
    });
    let late: ((result: ToolExecuteResult) => void) | undefined;
    const received: ToolExecuteResult[] = [];
    const bindings = createHostToolBindings({
      identity,
      tools,
      service: {
        version: 1,
        invoke: async (request, _signal, events) => {
          const onResult = events?.onResult;
          expect(request.identity.agentId).toBe('agent');
          expect(request.executingAgentId).toBe('child-agent');
          late = onResult;
          onResult?.({
            toolCallId: 'fast',
            status: 'success',
            content: 'fast',
            received_at: 1,
          });
          onResult?.({
            toolCallId: 'fast',
            status: 'success',
            content: 'fast',
          });
          first();
          await held;
          return [
            { toolCallId: 'slow', status: 'success', content: 'slow' },
            { toolCallId: 'fast', status: 'success', content: 'fast' },
          ];
        },
      },
    });
    const result = new Promise<ToolExecuteResult[]>((resolve, reject) => {
      void bindings.customHandlers[GraphEvents.ON_TOOL_EXECUTE].handle(
        GraphEvents.ON_TOOL_EXECUTE,
        {
          agentId: 'child-agent',
          toolCalls: ['fast', 'slow'].map((id) => ({
            id,
            name: 'echo',
            args: {},
          })),
          onResult: (r: ToolExecuteResult) => received.push(r),
          resolve,
          reject,
        }
      );
    });
    await ready;
    expect(received).toEqual([
      { toolCallId: 'fast', status: 'success', content: 'fast' },
    ]);
    release();
    expect((await result).map((r) => r.toolCallId)).toEqual(['fast', 'slow']);
    late?.({ toolCallId: 'slow', status: 'success', content: 'late' });
    expect(received).toHaveLength(1);
  });
  it('does not enable early completion when the SDK has gated it off', async () => {
    await dispatch(
      {
        version: 1,
        invoke: async (_request, _signal, events) => {
          const onResult = events?.onResult;
          expect(onResult).toBeUndefined();
          return [{ toolCallId: 'one', status: 'success', content: 'ok' }];
        },
      },
      [{ id: 'one', name: 'echo', args: {} }]
    );
  });
  it.each([false, true])(
    'retains local and explicitly transported session refreshes even on failure=%s',
    async (fail) => {
      for (const remote of [false, true]) {
        const call = {
          id: 'one',
          name: 'echo',
          args: {},
          codeSessionContext: { session_id: 'old' },
        };
        const bindings = createHostToolBindings({
          identity,
          tools,
          service: {
            version: 1,
            invoke: async (request, _signal, events) => {
              const context = {
                codeSessionContext: { session_id: 'refreshed' },
              };
              if (remote)
                events?.onSessionUpdate({ toolCallId: 'one', context });
              else Object.assign(request.calls[0].call, context);
              if (fail) throw new Error('failed after refresh');
              return [{ toolCallId: 'one', status: 'success', content: 'ok' }];
            },
          },
        });
        const result = new Promise<ToolExecuteResult[]>((resolve, reject) => {
          void bindings.customHandlers[GraphEvents.ON_TOOL_EXECUTE].handle(
            GraphEvents.ON_TOOL_EXECUTE,
            { toolCalls: [call], resolve, reject }
          );
        });
        if (fail)
          await expect(result).rejects.toThrow('Host tool execution failed');
        else await result;
        expect(call.codeSessionContext.session_id).toBe('refreshed');
      }
    }
  );
  describe.each(['direct', 'http'] as const)('%s service', (mode) => {
    it.each(['approve', 'edit', 'reject'] as const)(
      'uses native SDK approval %s and a real graph',
      async (decision) => {
        const invocations: HostToolRequest[] = [];
        const execute = async (
          request: HostToolRequest
        ): Promise<ToolExecuteResult[]> => {
          invocations.push(request);
          return request.calls.map(({ call }) => ({
            toolCallId: call.id,
            status: 'success',
            content: `echo:${call.args.text}`,
          }));
        };
        const server = createServer(async (req, res) => {
          let body = '';
          for await (const chunk of req) body += chunk.toString();
          res.setHeader('content-type', 'application/json');
          res.end(
            JSON.stringify(await execute(JSON.parse(body) as HostToolRequest))
          );
        });
        if (mode === 'http')
          await new Promise<void>((resolve) =>
            server.listen(0, '127.0.0.1', resolve)
          );
        try {
          const httpInvoke: HostToolService['invoke'] = async (
            request,
            signal
          ) => {
            const response = await fetch(
              `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
              { method: 'POST', body: JSON.stringify(request), signal }
            );
            return (await response.json()) as ToolExecuteResult[];
          };
          const service: HostToolService = {
            version: 1,
            invoke: mode === 'direct' ? execute : httpInvoke,
          };
          const bindings = createHostToolBindings({ identity, tools, service });
          const hooks = new HookRegistry();
          hooks.register('PreToolUse', {
            hooks: [async () => ({ decision: 'ask' })],
          });
          const run = await Run.create({
            runId: identity.runId,
            tokenCounter: () => 1,
            returnContent: true,
            skipCleanup: true,
            hooks,
            humanInTheLoop: { enabled: true },
            customHandlers: bindings.customHandlers,
            graphConfig: {
              type: 'standard',
              llmConfig: { provider: Providers.OPENAI },
              toolDefinitions: bindings.toolDefinitions,
              compileOptions: { checkpointer: new MemorySaver() },
            },
          });
          run.Graph!.overrideTestModel(['calling echo'], 5, [
            {
              id: 'echo-1',
              name: 'echo',
              args: { text: 'original' },
              type: 'tool_call',
            },
          ]);
          await run.processStream(
            { messages: [new HumanMessage('echo')] },
            caller
          );
          expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
          expect(invocations).toHaveLength(0);
          const answer = {
            type: decision,
            ...(decision === 'edit'
              ? { updatedInput: { text: 'edited' } }
              : {}),
            ...(decision === 'reject' ? { reason: 'no' } : {}),
          };
          await run.resume({ 'echo-1': answer }, caller);
          expect(invocations).toHaveLength(decision === 'reject' ? 0 : 1);
          if (decision !== 'reject')
            expect(invocations[0].calls[0].call.args.text).toBe(
              decision === 'edit' ? 'edited' : 'original'
            );
        } finally {
          if (mode === 'http')
            await new Promise<void>((resolve) => server.close(() => resolve()));
        }
      }
    );
  });
});

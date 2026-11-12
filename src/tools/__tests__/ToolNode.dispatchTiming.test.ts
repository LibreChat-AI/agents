import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { RunnableLambda } from '@langchain/core/runnables';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { PreToolUseHookOutput } from '@/hooks';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { GraphEvents } from '@/common';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

const makeTool = (
  name: string,
  run = async (): Promise<string> => 'direct result'
) =>
  tool(run, {
    name,
    description: name,
    schema: z.object({}).passthrough(),
  }) as StructuredToolInterface;

const message = (calls: Array<{ id: string; name: string }>) =>
  new AIMessage({
    content: '',
    tool_calls: calls.map(({ id, name }) => ({ id, name, args: {} })),
  });

describe('tool dispatch timing', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('separates model preparation from two independently finishing host calls', async () => {
    let now = 4_000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    const order: string[] = [];
    const starts: t.ToolCallsDispatchedEvent[] = [];
    const ends: t.ToolCompleteEvent[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_CALLS_DISPATCHED) {
          starts.push(data as t.ToolCallsDispatchedEvent);
          order.push('dispatched');
        } else if (event === GraphEvents.ON_TOOL_EXECUTE) {
          order.push('host');
          const batch = data as t.ToolExecuteBatchRequest;
          now = 4_025;
          batch.onResult?.({
            toolCallId: 'fast',
            content: 'fast result',
            status: 'success',
          });
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
          now = 4_400;
          batch.resolve([
            { toolCallId: 'fast', content: 'fast result', status: 'success' },
            { toolCallId: 'slow', content: 'slow result', status: 'success' },
          ]);
        } else if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          ends.push((data as { result: t.ToolCompleteEvent }).result);
          order.push(`completed:${ends.at(-1)?.tool_call.id}`);
        }
      });

    const node = new ToolNode({
      tools: [makeTool('fast_tool'), makeTool('slow_tool')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([
        ['fast', 'step_fast'],
        ['slow', 'step_slow'],
      ]),
    });
    const result = (await node.invoke(
      {
        messages: [
          message([
            { id: 'fast', name: 'fast_tool' },
            { id: 'slow', name: 'slow_tool' },
          ]),
        ],
      },
      { configurable: { run_id: 'run_1' } }
    )) as { messages: ToolMessage[] };

    expect(starts).toEqual([
      {
        dispatched_at: 4_000,
        runId: 'run_1',
        toolCalls: [
          { id: 'fast', name: 'fast_tool', stepId: 'step_fast' },
          { id: 'slow', name: 'slow_tool', stepId: 'step_slow' },
        ],
      },
    ]);
    expect(order.slice(0, 2)).toEqual(['dispatched', 'host']);
    expect(ends).toEqual([
      expect.objectContaining({
        tool_call: expect.objectContaining({ id: 'fast' }),
        completed_at: 4_025,
      }),
      expect.objectContaining({
        tool_call: expect.objectContaining({ id: 'slow' }),
        completed_at: 4_400,
      }),
    ]);
    expect(
      ends.map((end) => (end.completed_at ?? 0) - starts[0].dispatched_at)
    ).toEqual([25, 400]);
    expect(result.messages).toHaveLength(2);
  });

  it('stamps the fallback host result at resolve, not after handler cleanup', async () => {
    let now = 1_000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    const completions: t.ToolCompleteEvent[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          now = 1_200;
          (data as t.ToolExecuteBatchRequest).resolve([
            { toolCallId: 'fast', content: 'done', status: 'success' },
          ]);
          await Promise.resolve();
          now = 8_000;
        }
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push((data as { result: t.ToolCompleteEvent }).result);
        }
      }
    );
    const node = new ToolNode({
      tools: [makeTool('fast_tool')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['fast', 'step_fast']]),
    });
    await node.invoke({ messages: [message([{ id: 'fast', name: 'fast_tool' }])] });
    expect(completions).toHaveLength(1);
    expect(completions[0].completed_at).toBe(1_200);
  });

  it('does not announce dispatch for a call denied before host execution', async () => {
    const starts: t.ToolCallsDispatchedEvent[] = [];
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => ({
          decision: input.toolName === 'blocked' ? 'deny' : 'allow',
        }),
      ],
    });
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_CALLS_DISPATCHED) {
          starts.push(data as t.ToolCallsDispatchedEvent);
        }
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          const batch = data as t.ToolExecuteBatchRequest;
          expect(batch.toolCalls.map(({ id }) => id)).toEqual(['allowed']);
          batch.resolve([
            { toolCallId: 'allowed', content: 'ok', status: 'success' },
          ]);
        }
      });
    const node = new ToolNode({
      tools: [makeTool('blocked'), makeTool('allowed')],
      hookRegistry: registry,
      eventDrivenMode: true,
      toolCallStepIds: new Map([
        ['blocked', 'step_blocked'],
        ['allowed', 'step_allowed'],
      ]),
    });
    const result = (await node.invoke({
      messages: [
        message([
          { id: 'blocked', name: 'blocked' },
          { id: 'allowed', name: 'allowed' },
        ]),
      ],
    })) as { messages: ToolMessage[] };

    expect(starts).toHaveLength(1);
    expect(starts[0].toolCalls.map(({ id }) => id)).toEqual(['allowed']);
    expect(result.messages.map((item) => item.status)).toEqual([
      'error',
      'success',
    ]);
  });

  it('marks an in-process tool just before invoking it, not while its hooks run', async () => {
    let now = 7_000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    const order: string[] = [];
    const starts: t.ToolCallsDispatchedEvent[] = [];
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          now = 7_100;
          order.push('approved');
          return { decision: 'allow' };
        },
      ],
    });
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_CALLS_DISPATCHED) {
          starts.push(data as t.ToolCallsDispatchedEvent);
          order.push('dispatched');
        }
      });
    const node = new ToolNode({
      tools: [
        makeTool('direct', async () => {
          order.push('invoked');
          return 'ok';
        }),
      ],
      hookRegistry: registry,
      eventDrivenMode: true,
      directToolNames: new Set(['direct']),
      toolCallStepIds: new Map([['direct_id', 'step_direct']]),
    });
    await node.invoke({
      messages: [message([{ id: 'direct_id', name: 'direct' }])],
    });

    expect(starts).toEqual([
      {
        dispatched_at: 7_100,
        toolCalls: [{ id: 'direct_id', name: 'direct', stepId: 'step_direct' }],
      },
    ]);
    expect(order).toEqual(['approved', 'dispatched', 'invoked']);
  });

  it('does not announce execution for an already aborted direct call', async () => {
    const starts: t.ToolCallsDispatchedEvent[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_CALLS_DISPATCHED) {
          starts.push(data as t.ToolCallsDispatchedEvent);
        }
      });
    const controller = new AbortController();
    controller.abort(new Error('stopped'));
    const invoke = jest.fn(async (): Promise<string> => 'should not run');
    const node = new ToolNode({
      tools: [makeTool('direct', invoke)],
      directToolNames: new Set(['direct']),
      eventDrivenMode: true,
    });
    const result = (await node.invoke(
      { messages: [message([{ id: 'call_direct', name: 'direct' }])] },
      { signal: controller.signal }
    )) as { messages: ToolMessage[] };
    expect(result.messages[0].status).toBe('error');
    expect(starts).toHaveLength(0);
    expect(invoke).not.toHaveBeenCalled();
  });

  it('delivers the argument-free handoff through a real custom-event callback', async () => {
    const observed: t.ToolCallsDispatchedEvent[] = [];
    const callback = BaseCallbackHandler.fromMethods({
      handleCustomEvent: (name, data): void => {
        if (name === GraphEvents.ON_TOOL_CALLS_DISPATCHED) {
          observed.push(data as t.ToolCallsDispatchedEvent);
        }
      },
    });
    callback.awaitHandlers = true;
    const node = new ToolNode({
      tools: [makeTool('direct')],
      eventDrivenMode: true,
      directToolNames: new Set(['direct']),
      toolCallStepIds: new Map([['call_direct', 'step_direct']]),
    });
    await RunnableLambda.from(async (_input: string, config) =>
      node.invoke(
        { messages: [message([{ id: 'call_direct', name: 'direct' }])] },
        config
      )
    ).invoke('run', {
      configurable: { run_id: 'run_live' },
      callbacks: [callback],
    });

    expect(observed).toEqual([
      expect.objectContaining({
        runId: 'run_live',
        toolCalls: [
          { id: 'call_direct', name: 'direct', stepId: 'step_direct' },
        ],
      }),
    ]);
  });
});

describe('call_tool dispatcher', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  const definitions = new Map<string, t.LCTool>([
    ['bound_tool', { name: 'bound_tool' }],
    [
      'late_tool',
      {
        name: 'late_tool',
        defer_loading: true,
        parameters: { type: 'object', properties: { q: { type: 'string' } } },
      },
    ],
  ]);

  const createNode = (): ToolNode =>
    new ToolNode({
      tools: [makeTool('bound_tool'), makeTool('late_tool')],
      eventDrivenMode: true,
      toolDefinitions: definitions,
      toolDispatch: true,
      toolCallStepIds: new Map([
        ['call_ok', 'step_ok'],
        ['call_bad', 'step_bad'],
      ]),
    });

  /** Args the step-completed event carried for a call, as the host would persist them. */
  const completedArgs = (
    dispatch: { mock: { calls: unknown[][] } },
    id: string
  ): unknown => {
    type Completed = { result: { tool_call?: { id: string; args: unknown } } };
    const call = dispatch.mock.calls.find(
      ([event, data]) =>
        event === GraphEvents.ON_RUN_STEP_COMPLETED &&
        (data as Completed).result.tool_call?.id === id
    );
    const args = (call?.[1] as Completed | undefined)?.result.tool_call?.args;
    return typeof args === 'string' ? JSON.parse(args) : args;
  };

  const invoke = (
    node: ToolNode,
    id: string,
    args: Record<string, unknown>,
    name = 'call_tool'
  ): Promise<{ messages: ToolMessage[] }> =>
    node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [{ id, name, args }],
        }),
      ],
    }) as Promise<{ messages: ToolMessage[] }>;

  it('runs the target on the host and answers under the dispatcher call', async () => {
    const requests: t.ToolCallRequest[] = [];
    const completions: t.ProcessedToolCall[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          const batch = data as t.ToolExecuteBatchRequest;
          requests.push(...batch.toolCalls);
          batch.resolve([
            { toolCallId: 'call_ok', content: 'found', status: 'success' },
          ]);
        } else if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(
            (data as { result: t.ToolCompleteEvent }).result.tool_call
          );
        }
      });
    const dispatcherArgs = { name: 'late_tool', arguments: '{"q":"bolts"}' };

    const result = await invoke(createNode(), 'call_ok', dispatcherArgs);

    expect(requests).toEqual([
      expect.objectContaining({
        id: 'call_ok',
        name: 'late_tool',
        args: { q: 'bolts' },
      }),
    ]);
    expect(result.messages[0]).toMatchObject({
      name: 'call_tool',
      tool_call_id: 'call_ok',
      content: 'found',
    });
    expect(completions).toEqual([
      expect.objectContaining({
        id: 'call_ok',
        name: 'call_tool',
        args: JSON.stringify(dispatcherArgs),
        dispatched: { name: 'late_tool', args: '{"q":"bolts"}' },
      }),
    ]);
  });

  it('answers a call to a tool that is not deferred with an error and does not run it', async () => {
    const execute = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockResolvedValue(undefined);

    const result = await invoke(createNode(), 'call_bad', {
      name: 'bound_tool',
    });

    expect(result.messages[0]).toMatchObject({
      name: 'call_tool',
      tool_call_id: 'call_bad',
      status: 'error',
    });
    expect(result.messages[0].content).toContain(
      'not available through call_tool'
    );
    expect(
      execute.mock.calls.some(([event]) => event === GraphEvents.ON_TOOL_EXECUTE)
    ).toBe(false);
  });

  it('stores a direct call to a deferred tool as a dispatcher call', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          const batch = data as t.ToolExecuteBatchRequest;
          batch.resolve([
            { toolCallId: 'call_ok', content: 'found', status: 'success' },
          ]);
        }
      });
    const ai = new AIMessage({
      id: 'ai_1',
      content: [
        { type: 'tool_use', id: 'call_ok', name: 'late_tool', input: { q: 'x' } },
      ],
      additional_kwargs: {
        tool_calls: [
          {
            id: 'call_ok',
            type: 'function',
            function: { name: 'late_tool', arguments: '{"q":"x"}' },
          },
        ],
      },
      response_metadata: {
        output: [
          {
            type: 'function_call',
            id: 'fc_1',
            call_id: 'call_ok',
            name: 'late_tool',
            arguments: '{"q":"x"}',
          },
        ],
      },
      tool_calls: [{ id: 'call_ok', name: 'late_tool', args: { q: 'x' } }],
    });

    const result = (await createNode().invoke({ messages: [ai] })) as {
      messages: Array<AIMessage | ToolMessage>;
    };

    const [rewritten, answer] = result.messages as [AIMessage, ToolMessage];
    const dispatcherArgs = { name: 'late_tool', arguments: '{"q":"x"}' };
    expect(rewritten.tool_calls?.[0]).toMatchObject({
      id: 'call_ok',
      name: 'call_tool',
      args: dispatcherArgs,
    });
    expect(rewritten.content).toEqual([
      {
        type: 'tool_use',
        id: 'call_ok',
        name: 'call_tool',
        input: dispatcherArgs,
      },
    ]);
    expect(rewritten.additional_kwargs.tool_calls?.[0].function).toEqual({
      name: 'call_tool',
      arguments: JSON.stringify(dispatcherArgs),
    });
    expect(rewritten.response_metadata.output).toEqual([
      {
        type: 'function_call',
        id: 'fc_1',
        call_id: 'call_ok',
        name: 'call_tool',
        arguments: JSON.stringify(dispatcherArgs),
      },
    ]);
    expect(answer).toMatchObject({ name: 'call_tool', content: 'found' });
  });

  it('refuses a deferred target that canDispatch rejects without running it', async () => {
    const execute = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockResolvedValue(undefined);
    const node = new ToolNode({
      tools: [makeTool('bound_tool'), makeTool('late_tool')],
      eventDrivenMode: true,
      toolDefinitions: definitions,
      toolDispatch: { canDispatch: (name) => name !== 'late_tool' },
      toolCallStepIds: new Map([['call_bad', 'step_bad']]),
    });

    const result = await invoke(node, 'call_bad', {
      name: 'late_tool',
      arguments: '{"q":"x"}',
    });

    expect(result.messages[0]).toMatchObject({
      name: 'call_tool',
      tool_call_id: 'call_bad',
      status: 'error',
    });
    expect(completedArgs(execute, 'call_bad')).toEqual({
      name: 'late_tool',
      arguments: '{"q":"x"}',
    });
    expect(result.messages[0].content).toContain(
      'not available through call_tool'
    );
    expect(
      execute.mock.calls.some(([event]) => event === GraphEvents.ON_TOOL_EXECUTE)
    ).toBe(false);
  });

  it('refuses a disabled tool without running it', async () => {
    const execute = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockResolvedValue(undefined);
    const node = new ToolNode({
      tools: [makeTool('off_tool')],
      eventDrivenMode: true,
      toolDefinitions: new Map([
        ['off_tool', { name: 'off_tool', disabled: { reason: 'switched off' } }],
      ]),
      toolCallStepIds: new Map([['call_bad', 'step_bad']]),
    });

    const result = await invoke(node, 'call_bad', { q: 'x' }, 'off_tool');

    expect(result.messages[0]).toMatchObject({
      name: 'off_tool',
      status: 'error',
    });
    expect(result.messages[0].content).toContain('switched off');
    expect(completedArgs(execute, 'call_bad')).toEqual({ q: 'x' });
    expect(
      execute.mock.calls.some(([event]) => event === GraphEvents.ON_TOOL_EXECUTE)
    ).toBe(false);
  });

  it('completes a failed dispatched call under the dispatcher call, not through the error handler', async () => {
    const completions: t.ProcessedToolCall[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(
            (data as { result: t.ToolCompleteEvent }).result.tool_call
          );
        }
      });
    const errorHandler = jest.fn(async () => true);
    const node = new ToolNode({
      tools: [
        makeTool('late_tool', async () => {
          throw new Error('boom');
        }),
      ],
      toolDefinitions: definitions,
      toolDispatch: true,
      errorHandler:
        errorHandler as unknown as t.ToolNodeConstructorParams['errorHandler'],
      toolCallStepIds: new Map([['call_ok', 'step_ok']]),
    });

    const result = await invoke(node, 'call_ok', {
      name: 'late_tool',
      arguments: '{"q":"x"}',
    });

    expect(errorHandler).not.toHaveBeenCalled();
    expect(result.messages[0]).toMatchObject({
      name: 'call_tool',
      status: 'error',
    });
    expect(completions).toEqual([
      expect.objectContaining({
        id: 'call_ok',
        name: 'call_tool',
        dispatched: { name: 'late_tool', args: '{"q":"x"}' },
      }),
    ]);
  });

  it('unwraps and refuses calls that arrive through Send like those in an assistant message', async () => {
    const requests: t.ToolCallRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          const batch = data as t.ToolExecuteBatchRequest;
          requests.push(...batch.toolCalls);
          batch.resolve(
            batch.toolCalls.map((call) => ({
              toolCallId: call.id,
              content: 'found',
              status: 'success' as const,
            }))
          );
        }
      });
    const node = new ToolNode({
      tools: [makeTool('late_tool'), makeTool('off_tool')],
      eventDrivenMode: true,
      toolDefinitions: new Map([
        ...definitions,
        ['off_tool', { name: 'off_tool', disabled: { reason: 'switched off' } }],
      ]),
      toolDispatch: true,
    });
    const send = (
      id: string,
      name: string,
      args: Record<string, unknown>
    ): Promise<{ messages: ToolMessage[] }> =>
      node.invoke({
        messages: [],
        lg_tool_call: { id, name, args },
      }) as Promise<{ messages: ToolMessage[] }>;

    const dispatched = await send('call_ok', 'call_tool', {
      name: 'late_tool',
      arguments: '{"q":"x"}',
    });
    const disabled = await send('call_off', 'off_tool', {});

    expect(requests).toEqual([
      expect.objectContaining({ id: 'call_ok', name: 'late_tool' }),
    ]);
    expect(dispatched.messages[0]).toMatchObject({
      name: 'call_tool',
      content: 'found',
    });
    expect(disabled.messages[0]).toMatchObject({
      name: 'off_tool',
      status: 'error',
    });
    expect(disabled.messages[0].content).toContain('switched off');
  });
});

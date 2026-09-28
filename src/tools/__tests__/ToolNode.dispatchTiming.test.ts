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

import { z } from 'zod';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ToolResultProtection } from '@/protection/toolResult';
import { ToolOutputReferenceRegistry } from '../toolOutputReferences';
import { PreparedSubagents } from '../preparedSubagents';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';
import { Constants } from '@/common';

const call: ToolCall = {
  id: 'child-call',
  name: Constants.SUBAGENT,
  args: { description: 'Research this', subagent_type: 'researcher' },
};
const config = { configurable: { run_id: 'run', thread_id: 'thread' } };

function fixture(
  hookRegistry?: HookRegistry,
  nonCooperative = false,
  toolResultProtection?: ToolResultProtection
) {
  let starts = 0;
  let finish!: (value: string) => void;
  let fail!: (error: Error) => void;
  let onStarted!: () => void;
  const started = new Promise<void>((resolve) => {
    onStarted = resolve;
  });
  const result = new Promise<string>((resolve, reject) => {
    finish = resolve;
    fail = reject;
  });
  const func = async () => {
    starts++;
    onStarted();
    return result;
  };
  const params = {
    name: Constants.SUBAGENT,
    description: 'Delegate research',
    schema: z.object({ description: z.string(), subagent_type: z.string() }),
  };
  // tool() itself races cancellation. Use the direct class to model an
  // invocation that remains pending after its signal aborts.
  const child = nonCooperative
    ? new DynamicStructuredTool({ ...params, func })
    : tool(func, params);
  const prepared = new PreparedSubagents();
  const registry = new ToolOutputReferenceRegistry();
  const node = new ToolNode({
    tools: [child],
    executingAgentId: 'parent',
    preparedSubagents: prepared,
    eventDrivenMode: true,
    directToolNames: new Set([Constants.SUBAGENT]),
    eagerEventToolExecution: { enabled: true },
    toolOutputRegistry: registry,
    hookRegistry,
    toolResultProtection,
  });
  return {
    node,
    prepared,
    started,
    finish,
    fail,
    registry,
    starts: () => starts,
  };
}

describe('ToolNode prepared subagent adoption', () => {
  it('retains capacity until a cancellation-ignoring tool actually settles', async () => {
    const f = fixture(undefined, true);
    const controller = new AbortController();
    const scoped = { ...config, signal: controller.signal };
    f.prepared.begin('old');
    AsyncLocalStorageProviderSingleton.runWithConfig(scoped, () =>
      f.node.prestartSubagent(call, 'old', scoped, 1)
    );
    await f.started;
    controller.abort(new Error('cancelled'));
    f.prepared.clear();
    await new Promise<void>((resolve) => setImmediate(resolve));
    f.prepared.begin('new');
    expect(
      f.prepared.start(
        'new',
        'parent',
        { ...call, id: 'next' },
        1,
        async () => 'next'
      )
    ).toBe(false);
    f.finish('late output');
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(
      f.prepared.start(
        'new',
        'parent',
        { ...call, id: 'next' },
        1,
        async () => 'next'
      )
    ).toBe(true);
    f.prepared.clear();
  });

  it('keeps raw failures out of the dispatch chain while preserving adoption errors', async () => {
    const f = fixture();
    const outputs: unknown[] = [];
    const errors: unknown[] = [];
    f.prepared.begin('attempt');
    f.node.prestartSubagent(
      call,
      'attempt',
      {
        ...config,
        callbacks: [
          {
            handleChainEnd: (output: unknown) => {
              outputs.push(output);
            },
            handleChainError: (error: unknown) => {
              errors.push(error);
            },
          },
        ],
      },
      1
    );
    await f.started;
    f.prepared.finish('attempt', [call]);
    f.fail(new Error('sensitive failure detail'));
    await expect(f.prepared.take('parent', call)).rejects.toThrow(
      'sensitive failure detail'
    );
    expect(outputs).toEqual([{ status: 'failed' }]);
    expect(errors).toEqual([]);
  });

  it('parents each early tool beneath its own dispatch chain', async () => {
    const f = fixture();
    const chains: Array<{ id: string; name?: string; input: unknown }> = [];
    const parents: Array<string | undefined> = [];
    const chainOutputs = new Map<string, unknown>();
    const tracedConfig = {
      ...config,
      runId: '00000000-0000-4000-8000-000000000001',
      callbacks: [
        {
          handleChainStart: (
            _chain: unknown,
            input: unknown,
            id: string,
            _parent?: string,
            _tags?: string[],
            _metadata?: Record<string, unknown>,
            _type?: string,
            name?: string
          ) => {
            chains.push({ id, name, input });
          },
          handleChainEnd: (output: unknown, id: string) => {
            chainOutputs.set(id, output);
          },
          handleToolStart: (
            _tool: unknown,
            _input: string,
            _id: string,
            parent?: string
          ) => {
            parents.push(parent);
          },
        },
      ],
    };
    f.prepared.begin('attempt');
    f.node.prestartSubagent(call, 'attempt', tracedConfig, 4);
    await f.started;
    expect(parents).toHaveLength(1);
    const dispatch = chains.find((chain) => chain.id === parents[0]);
    expect(dispatch?.name).toBe('tools=parent');
    expect(dispatch?.id).not.toBe(tracedConfig.runId);
    expect(dispatch?.input).toEqual({
      messages: [expect.objectContaining({ tool_calls: [call] })],
    });
    f.prepared.finish('attempt', [call]);
    f.finish('research complete');
    const result = await f.node.invoke(
      [new AIMessage({ content: '', tool_calls: [call] })],
      tracedConfig
    );
    expect(result[0].content).toBe('research complete');
    expect(chainOutputs.get(dispatch!.id)).toEqual({ status: 'completed' });
    expect(parents).toHaveLength(1);
  });

  it('runs once before the batch and still registers the final output reference', async () => {
    const f = fixture();
    f.prepared.begin('attempt');
    f.node.prestartSubagent(call, 'attempt', config, 4);
    await f.started;
    expect(f.starts()).toBe(1);
    f.prepared.finish('attempt', [call]);
    const output = f.node.invoke(
      [new AIMessage({ content: '', tool_calls: [call] })],
      config
    );
    f.finish('research complete');
    const messages = (await output) as ToolMessage[];
    expect(f.starts()).toBe(1);
    expect(messages[0].content).toBe('research complete');
    expect(messages[0].additional_kwargs._refKey).toBeDefined();
    expect(f.node.getToolUsageCounts().get(Constants.SUBAGENT)).toBe(1);
  });

  it('leaves calls behind parent approval hooks on the normal path', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [async () => ({ decision: 'deny' as const })],
    });
    const f = fixture(hooks);
    f.prepared.begin('attempt');
    f.node.prestartSubagent(call, 'attempt', config, 4);
    await Promise.resolve();
    expect(f.starts()).toBe(0);
    f.prepared.finish('attempt', [call]);
    const output = (await f.node.invoke(
      [new AIMessage({ content: '', tool_calls: [call] })],
      config
    )) as ToolMessage[];
    expect(output[0].status).toBe('error');
    expect(f.starts()).toBe(0);
  });

  it('leaves a subagent tool with protected results on the normal path', async () => {
    const f = fixture(undefined, false, {
      version: 1,
      toolNames: [Constants.SUBAGENT],
      timeoutMs: 1000,
      maxAttemptBytes: 4096,
      maxBufferedBytes: 4096,
      classify: () => 'prose',
      inspect: () => ({
        version: 1,
        ok: true,
        value: { content: 'inspected', replacements: 1, categories: [] },
      }),
    });
    /** Earlier adoptions stamp their token on the shared call object. */
    const call: ToolCall = {
      id: 'protected-call',
      name: Constants.SUBAGENT,
      args: { description: 'Research this', subagent_type: 'researcher' },
    };
    f.prepared.begin('attempt');
    f.node.prestartSubagent(call, 'attempt', config, 4);
    const startedEarly = await Promise.race([
      f.started.then(() => true),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 50)),
    ]);
    expect(startedEarly).toBe(false);
    f.prepared.finish('attempt', [call]);
    const pending = f.node.invoke(
      [new AIMessage({ content: '', tool_calls: [call] })],
      config
    );
    await f.started;
    f.finish('raw child answer');
    const output = (await pending) as ToolMessage[];
    expect(f.starts()).toBe(1);
    expect(output[0].content).toBe('inspected');
  });

  it.each([{ run_in_background: true }, { subagent_thread_id: 'thread' }])(
    'never speculatively starts detached or continued work: %p',
    async (args) => {
      const f = fixture();
      f.prepared.begin('attempt');
      f.node.prestartSubagent(
        { ...call, args: { ...call.args, ...args } },
        'attempt',
        config,
        4
      );
      await Promise.resolve();
      expect(f.starts()).toBe(0);
      f.prepared.clear();
    }
  );
  it('fails closed if a parent policy hook is registered after admission', async () => {
    const hooks = new HookRegistry();
    const f = fixture(hooks);
    f.prepared.begin('attempt');
    f.node.prestartSubagent(call, 'attempt', config, 4);
    await f.started;
    f.prepared.finish('attempt', [call]);
    hooks.register('PreToolUse', {
      hooks: [async () => ({ decision: 'deny' as const })],
    });
    await expect(
      f.node.invoke(
        [new AIMessage({ content: '', tool_calls: [call] })],
        config
      )
    ).rejects.toThrow('Tool hooks changed');
    f.finish('cancelled child settles');
    expect(f.starts()).toBe(1);
  });
});

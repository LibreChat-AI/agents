import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import {
  END,
  START,
  Command,
  StateGraph,
  MemorySaver,
  isInterrupted,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { BaseMessage, ToolMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { PreToolUseHookInput, PreToolUseHookOutput } from '@/hooks';
import type * as t from '@/types';
import {
  TOOL_DISPATCHER_NAME,
  resolveDispatchedToolCall,
  rewriteDispatchedToolCalls,
  createToolDispatcherDefinition,
} from '../toolDispatch';
import { createToolSearch } from '../ToolSearch';
import {
  TOOL_BATCH_REPLAY_KEY,
  getToolBatchReplayState,
} from '../toolBatchReplay';
import { HookRegistry, createToolPolicyHook } from '@/hooks';
import { AgentContext } from '@/agents/AgentContext';
import { GraphEvents, Providers } from '@/common';
import * as events from '@/utils/events';
import { ToolNode } from '../ToolNode';

const DISPATCH = { name: TOOL_DISPATCHER_NAME };

const QUERY_SCHEMA: t.JsonSchemaType = {
  type: 'object',
  properties: {
    query: { type: 'string' },
    limit: { type: 'integer' },
  },
  required: ['query'],
};

const DEFINITIONS: t.LCTool[] = [
  {
    name: 'lookup_order',
    description: 'Bound tool.',
    parameters: QUERY_SCHEMA,
  },
  {
    name: 'search_wiki',
    description: 'Made available mid-conversation.',
    parameters: QUERY_SCHEMA,
    defer_loading: true,
  },
  {
    name: 'run_script',
    description: 'Code execution only.',
    parameters: QUERY_SCHEMA,
    defer_loading: true,
    allowed_callers: ['code_execution'],
  },
  {
    name: 'old_search',
    description: 'Switched off by the user.',
    parameters: QUERY_SCHEMA,
    disabled: { reason: 'the user turned it off' },
  },
];

const registry = new Map(DEFINITIONS.map((def) => [def.name, def]));

function dispatcherCall(
  name: unknown,
  args?: unknown,
  id = 'call_d'
): { id: string; name: string; args: Record<string, unknown> } {
  return {
    id,
    name: TOOL_DISPATCHER_NAME,
    args: { name, ...(args !== undefined && { arguments: args }) },
  };
}

function errorOf(
  resolution: ReturnType<typeof resolveDispatchedToolCall>
): string {
  if (!('error' in resolution)) {
    throw new Error('expected a refusal');
  }
  return resolution.error;
}

describe('resolveDispatchedToolCall', () => {
  it('unwraps a dispatcher call to its target under the same id', () => {
    const resolution = resolveDispatchedToolCall(
      dispatcherCall('search_wiki', '{"query":"bolts","limit":"3"}'),
      registry,
      DISPATCH
    );
    expect(resolution).toEqual({
      call: {
        id: 'call_d',
        name: 'search_wiki',
        args: { query: 'bolts', limit: 3 },
      },
      dispatcher: TOOL_DISPATCHER_NAME,
    });
  });

  it('accepts arguments given as an object', () => {
    const resolution = resolveDispatchedToolCall(
      dispatcherCall('search_wiki', { query: 'nuts' }),
      registry,
      DISPATCH
    );
    expect('call' in resolution && resolution.call.args).toEqual({
      query: 'nuts',
    });
  });

  it('passes ordinary calls through untouched', () => {
    const call = { id: 'c', name: 'lookup_order', args: { query: '42' } };
    expect(resolveDispatchedToolCall(call, registry, DISPATCH)).toEqual({
      call,
    });
    expect(resolveDispatchedToolCall(call, registry)).toEqual({ call });
  });

  it('does not unwrap when the host did not enable the dispatcher', () => {
    const call = dispatcherCall('search_wiki', '{"query":"x"}');
    expect(resolveDispatchedToolCall(call, registry)).toEqual({ call });
    const direct = { id: 'c', name: 'search_wiki', args: { query: 'x' } };
    expect(resolveDispatchedToolCall(direct, registry)).toEqual({
      call: direct,
    });
  });

  it('runs a direct call to a dispatchable tool as a dispatched call', () => {
    expect(
      resolveDispatchedToolCall(
        { id: 'c', name: 'search_wiki', args: { query: 'x', limit: '2' } },
        registry,
        DISPATCH
      )
    ).toEqual({
      call: { id: 'c', name: 'search_wiki', args: { query: 'x', limit: 2 } },
      dispatcher: TOOL_DISPATCHER_NAME,
      rewrite: {
        name: TOOL_DISPATCHER_NAME,
        args: { name: 'search_wiki', arguments: '{"query":"x","limit":2}' },
      },
    });
    expect(
      errorOf(
        resolveDispatchedToolCall(
          { id: 'c', name: 'search_wiki', args: {} },
          registry,
          DISPATCH
        )
      )
    ).toContain('Invalid arguments for "search_wiki"');
  });

  it('rewrites every provider-native copy of a rewritten call', () => {
    const rewrite = {
      name: TOOL_DISPATCHER_NAME,
      args: { name: 'search_wiki', arguments: '{"query":"x"}' },
    };
    const rewritten = rewriteDispatchedToolCalls(
      new AIMessage({
        id: 'msg_1',
        content: [
          {
            type: 'tool_use',
            id: 'c',
            name: 'search_wiki',
            input: { query: 'x' },
          },
        ],
        additional_kwargs: {
          tool_calls: [
            {
              id: 'c',
              type: 'function',
              function: { name: 'search_wiki', arguments: '{"query":"x"}' },
            },
          ],
        },
        response_metadata: {
          output: [
            {
              type: 'function_call',
              id: 'fc_1',
              call_id: 'c',
              name: 'search_wiki',
              arguments: '{"query":"x"}',
            },
          ],
        },
        tool_calls: [{ id: 'c', name: 'search_wiki', args: { query: 'x' } }],
      }),
      new Map([['c', rewrite]])
    );
    const serialized = JSON.stringify([
      rewritten.content,
      rewritten.additional_kwargs,
      rewritten.response_metadata,
      rewritten.tool_calls,
    ]);
    expect(serialized.match(/"name":"call_tool"/g)).toHaveLength(4);
    expect(rewritten.id).toBe('msg_1');
    expect(rewritten.tool_calls).toEqual([
      expect.objectContaining({ id: 'c', ...rewrite }),
    ]);
    expect(rewritten.content).toEqual([
      {
        type: 'tool_use',
        id: 'c',
        name: TOOL_DISPATCHER_NAME,
        input: rewrite.args,
      },
    ]);
  });

  it('refuses unknown targets, bound tools, and code-execution-only tools', () => {
    expect(
      errorOf(
        resolveDispatchedToolCall(dispatcherCall('nope'), registry, DISPATCH)
      )
    ).toBe('Tool "nope" is not available through call_tool.');
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall('lookup_order', '{"query":"1"}'),
          registry,
          DISPATCH
        )
      )
    ).toBe(
      'Tool "lookup_order" is not available through call_tool. It is in your tool list: call it directly.'
    );
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall('run_script', '{"query":"1"}'),
          registry,
          DISPATCH
        )
      )
    ).toContain('not available through call_tool');
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall(TOOL_DISPATCHER_NAME),
          registry,
          DISPATCH
        )
      )
    ).toContain('not available');
    expect(
      errorOf(resolveDispatchedToolCall(dispatcherCall(''), registry, DISPATCH))
    ).toContain('requires "name"');
  });

  it('refuses disabled tools whether called directly or through the dispatcher', () => {
    const expected =
      'Tool "old_search" is disabled in this conversation: the user turned it off';
    expect(
      errorOf(
        resolveDispatchedToolCall(
          { id: 'c', name: 'old_search', args: { query: 'x' } },
          registry
        )
      )
    ).toBe(expected);
    const deferredOff = new Map(registry).set('old_search', {
      ...registry.get('old_search')!,
      defer_loading: true,
    });
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall('old_search', '{"query":"x"}'),
          deferredOff,
          DISPATCH
        )
      )
    ).toBe(expected);
  });

  it('refuses arguments that are not a JSON object or do not match the target schema', () => {
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall('search_wiki', '[1]'),
          registry,
          DISPATCH
        )
      )
    ).toBe('The arguments for "search_wiki" must be a JSON object.');
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall('search_wiki', '{"limit":"many"}'),
          registry,
          DISPATCH
        )
      )
    ).toBe(
      'Invalid arguments for "search_wiki": #: Instance does not have required property "query".; #/limit: Instance type "string" is invalid. Expected "integer".'
    );
  });

  it('reads the non-standard float type as number', () => {
    const floats = new Map([
      [
        'measure',
        {
          name: 'measure',
          defer_loading: true,
          parameters: {
            type: 'object',
            properties: { value: { type: 'float' } },
          },
        } as t.LCTool,
      ],
    ]);
    expect(
      'call' in
        resolveDispatchedToolCall(
          dispatcherCall('measure', '{"value":1.5}'),
          floats,
          DISPATCH
        )
    ).toBe(true);
  });
});

describe('resolveDispatchedToolCall with schemas the validator cannot apply', () => {
  const unusable = new Map<string, t.LCTool>([
    [
      'bad_pattern',
      {
        name: 'bad_pattern',
        defer_loading: true,
        parameters: {
          type: 'object',
          properties: {
            year: { type: 'string', pattern: '^(?P<year>\\d{4})$' },
          },
        },
      } as t.LCTool,
    ],
    [
      'bad_ref',
      {
        name: 'bad_ref',
        defer_loading: true,
        parameters: {
          type: 'object',
          properties: { item: { $ref: '#/$defs/missing' } },
        },
      } as unknown as t.LCTool,
    ],
  ]);

  it.each(['bad_pattern', 'bad_ref'])(
    'returns an error for %s instead of throwing',
    (target) => {
      const args = target === 'bad_ref' ? '{"item":{}}' : '{"year":"2024"}';
      const resolution = resolveDispatchedToolCall(
        dispatcherCall(target, args),
        unusable,
        DISPATCH
      );
      expect(errorOf(resolution)).toContain(
        `Invalid arguments for "${target}": the tool's schema could not be applied`
      );
    }
  );
});

describe('resolveDispatchedToolCall with a host target filter', () => {
  const allowed = {
    ...DISPATCH,
    canDispatch: (name: string) => name === 'search_wiki',
  };
  const extended = new Map(registry);
  extended.set('other_deferred', {
    name: 'other_deferred',
    parameters: QUERY_SCHEMA,
    defer_loading: true,
  });

  it('dispatches only the tools the host declared', () => {
    expect(
      'call' in
        resolveDispatchedToolCall(
          dispatcherCall('search_wiki', '{"query":"x"}'),
          extended,
          allowed
        )
    ).toBe(true);
    expect(
      errorOf(
        resolveDispatchedToolCall(
          dispatcherCall('other_deferred', '{"query":"x"}'),
          extended,
          allowed
        )
      )
    ).toBe('Tool "other_deferred" is not available through call_tool.');
  });

  it('refuses a direct call to an undeclared deferred tool', () => {
    expect(
      errorOf(
        resolveDispatchedToolCall(
          { id: 'c', name: 'other_deferred', args: { query: 'x' } },
          extended,
          allowed
        )
      )
    ).toBe('Tool "other_deferred" is not available through call_tool.');
  });
});

function createSchemaStub(name: string): StructuredToolInterface {
  return tool(async () => 'unused', {
    name,
    description: 'schema-only stub; host executes via ON_TOOL_EXECUTE',
    schema: z.object({ query: z.string() }),
  }) as unknown as StructuredToolInterface;
}

type Captured = {
  requests: t.ToolCallRequest[];
  completions: t.ProcessedToolCall[];
};

function captureEvents(output = 'found'): Captured {
  const captured: Captured = { requests: [], completions: [] };
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
        captured.completions.push(
          (data as { result: { tool_call: t.ProcessedToolCall } }).result
            .tool_call
        );
        return;
      }
      if (event !== GraphEvents.ON_TOOL_EXECUTE) {
        return;
      }
      const batch = data as t.ToolExecuteBatchRequest;
      captured.requests.push(...batch.toolCalls);
      batch.resolve(
        batch.toolCalls.map((call) => ({
          toolCallId: call.id,
          content: `${output}:${call.name}`,
          status: 'success' as const,
        }))
      );
    });
  return captured;
}

function capturePreToolUse(
  inputs: PreToolUseHookInput[],
  output: PreToolUseHookOutput = { decision: 'allow' }
): HookRegistry {
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', {
    hooks: [
      async (input): Promise<PreToolUseHookOutput> => {
        inputs.push(input);
        return output;
      },
    ],
  });
  return hooks;
}

function createEventNode(
  options: Partial<t.ToolNodeConstructorParams> = {}
): ToolNode {
  return new ToolNode({
    tools: DEFINITIONS.map((def) => createSchemaStub(def.name)),
    eventDrivenMode: true,
    agentId: 'agent-x',
    toolDefinitions: registry,
    toolDispatch: DISPATCH,
    toolCallStepIds: new Map(
      ['call_d', 'call_off', 'call_unknown', 'call_bad', 'call_ok'].map(
        (id) => [id, `step_${id}`]
      )
    ),
    ...options,
  });
}

function messagesOf(result: unknown): ToolMessage[] {
  return (result as { messages: ToolMessage[] }).messages;
}

describe('ToolNode tool dispatch (event-driven)', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('runs the target for hooks and the host, and answers the dispatcher call', async () => {
    const captured = captureEvents();
    const hookInputs: PreToolUseHookInput[] = [];
    const node = createEventNode({
      hookRegistry: capturePreToolUse(hookInputs),
    });

    const result = await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [dispatcherCall('search_wiki', '{"query":"bolts"}')],
        }),
      ],
    });

    expect(hookInputs).toHaveLength(1);
    expect(hookInputs[0]).toMatchObject({
      toolName: 'search_wiki',
      toolInput: { query: 'bolts' },
      toolUseId: 'call_d',
    });
    expect(captured.requests).toEqual([
      expect.objectContaining({
        id: 'call_d',
        name: 'search_wiki',
        args: { query: 'bolts' },
        dispatcher: TOOL_DISPATCHER_NAME,
      }),
    ]);
    const [message] = messagesOf(result);
    expect(message).toMatchObject({
      name: TOOL_DISPATCHER_NAME,
      tool_call_id: 'call_d',
      content: 'found:search_wiki',
    });
    expect(captured.completions).toEqual([
      expect.objectContaining({
        id: 'call_d',
        name: TOOL_DISPATCHER_NAME,
        args: JSON.stringify({
          name: 'search_wiki',
          arguments: '{"query":"bolts"}',
        }),
        output: 'found:search_wiki',
        dispatched: { name: 'search_wiki', args: '{"query":"bolts"}' },
      }),
    ]);
  });

  it('refuses disabled tools, unknown targets, and schema mismatches before any hook or the host', async () => {
    const captured = captureEvents();
    const hookInputs: PreToolUseHookInput[] = [];
    const node = createEventNode({
      hookRegistry: capturePreToolUse(hookInputs),
    });

    const result = await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'call_off', name: 'old_search', args: { query: 'x' } },
            dispatcherCall('nope', '{}', 'call_unknown'),
            dispatcherCall('search_wiki', '{"limit":1}', 'call_bad'),
            { id: 'call_ok', name: 'lookup_order', args: { query: '42' } },
          ],
        }),
      ],
    });

    expect(hookInputs.map((input) => input.toolName)).toEqual(['lookup_order']);
    expect(captured.requests.map((request) => request.name)).toEqual([
      'lookup_order',
    ]);
    const byId = new Map(
      messagesOf(result).map((message) => [message.tool_call_id, message])
    );
    expect(byId.get('call_ok')).toMatchObject({ status: 'success' });
    expect(byId.get('call_off')).toMatchObject({
      status: 'error',
      name: 'old_search',
      content: expect.stringContaining(
        'Tool "old_search" is disabled in this conversation'
      ),
    });
    expect(byId.get('call_unknown')).toMatchObject({
      status: 'error',
      name: TOOL_DISPATCHER_NAME,
      content: expect.stringContaining('"nope" is not available'),
    });
    expect(byId.get('call_bad')).toMatchObject({
      status: 'error',
      name: TOOL_DISPATCHER_NAME,
      content: expect.stringContaining('Invalid arguments for "search_wiki"'),
    });
    expect(captured.completions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'call_off', name: 'old_search' }),
      ])
    );
  });

  it('answers a call whose target schema cannot be applied with a tool error', async () => {
    captureEvents();
    const broken: t.LCTool = {
      name: 'broken_schema',
      defer_loading: true,
      parameters: {
        type: 'object',
        properties: { year: { type: 'string', pattern: '^(?P<y>\\d+)$' } },
      },
    };
    const node = createEventNode({
      toolDefinitions: new Map([...registry, ['broken_schema', broken]]),
    });

    const result = await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [dispatcherCall('broken_schema', '{"year":"1"}')],
        }),
      ],
    });

    expect(messagesOf(result)[0]).toMatchObject({
      status: 'error',
      name: TOOL_DISPATCHER_NAME,
      content: expect.stringContaining('schema could not be applied'),
    });
  });

  it('keeps today\'s behaviour when the dispatcher is not enabled', async () => {
    const captured = captureEvents();
    const node = createEventNode({ toolDispatch: undefined });

    await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [dispatcherCall('search_wiki', '{"query":"x"}')],
        }),
      ],
    });

    expect(captured.requests).toEqual([
      expect.objectContaining({ name: TOOL_DISPATCHER_NAME }),
    ]);
    expect(captured.requests[0].dispatcher).toBeUndefined();
  });
});

describe('ToolNode tool dispatch of direct calls to dispatchable tools', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('runs the target for hooks and rewrites the call into a dispatcher call', async () => {
    const captured = captureEvents();
    const hookInputs: PreToolUseHookInput[] = [];
    const node = createEventNode({
      hookRegistry: capturePreToolUse(hookInputs),
    });
    const args = { query: 'bolts' };

    const result = await node.invoke({
      messages: [
        new AIMessage({
          id: 'msg_direct',
          content: [
            {
              type: 'tool_use',
              id: 'call_d',
              name: 'search_wiki',
              input: args,
            },
          ],
          tool_calls: [{ id: 'call_d', name: 'search_wiki', args }],
        }),
      ],
    });

    expect(hookInputs[0]).toMatchObject({
      toolName: 'search_wiki',
      toolInput: args,
    });
    expect(captured.requests).toEqual([
      expect.objectContaining({
        name: 'search_wiki',
        dispatcher: TOOL_DISPATCHER_NAME,
      }),
    ]);
    const modelArgs = {
      name: 'search_wiki',
      arguments: JSON.stringify(args),
    };
    const [replacement, message] = (
      result as { messages: Array<AIMessage | ToolMessage> }
    ).messages;
    expect(replacement).toBeInstanceOf(AIMessage);
    expect(replacement).toMatchObject({
      id: 'msg_direct',
      tool_calls: [
        expect.objectContaining({
          id: 'call_d',
          name: TOOL_DISPATCHER_NAME,
          args: modelArgs,
        }),
      ],
      content: [
        {
          type: 'tool_use',
          id: 'call_d',
          name: TOOL_DISPATCHER_NAME,
          input: modelArgs,
        },
      ],
    });
    expect(message).toMatchObject({
      name: TOOL_DISPATCHER_NAME,
      tool_call_id: 'call_d',
    });
    expect(captured.completions).toEqual([
      expect.objectContaining({
        name: TOOL_DISPATCHER_NAME,
        args: JSON.stringify(modelArgs),
        dispatched: { name: 'search_wiki', args: JSON.stringify(args) },
      }),
    ]);
  });
});

describe('tool_search with a dispatcher', () => {
  const searchRegistry = new Map(DEFINITIONS.map((def) => [def.name, def]));

  async function search(toolDispatch?: string): Promise<string> {
    const toolSearch = createToolSearch({
      mode: 'local',
      toolRegistry: searchRegistry,
      toolDispatch,
    });
    const output = await toolSearch.invoke({
      id: 'call_search',
      name: 'tool_search',
      args: { query: 'wiki' },
      type: 'tool_call',
    });
    return String((output as ToolMessage).content);
  }

  it('lists full names and schemas and tells the model to use the dispatcher', async () => {
    const result = JSON.parse(await search(TOOL_DISPATCHER_NAME)) as {
      tools: Array<{ name: string; parameters?: unknown }>;
      call_with?: string;
    };
    expect(result.tools).toEqual([
      expect.objectContaining({
        name: 'search_wiki',
        parameters: QUERY_SCHEMA,
      }),
    ]);
    expect(result.call_with).toContain(`Call one with ${TOOL_DISPATCHER_NAME}`);
  });

  it('keeps today\'s output without a dispatcher', async () => {
    const result = JSON.parse(await search()) as Record<string, unknown>;
    expect(result.call_with).toBeUndefined();
    expect(JSON.stringify(result)).not.toContain('"parameters"');
  });
});

describe('ToolNode tool dispatch (in-graph tools)', () => {
  it('invokes the target tool instance with lifecycle hooks seeing the target', async () => {
    const executed: unknown[] = [];
    const searchWiki = tool(
      async (args: { query: string }) => {
        executed.push(args);
        return `wiki:${args.query}`;
      },
      {
        name: 'search_wiki',
        description: 'Made available mid-conversation.',
        schema: z.object({ query: z.string() }),
      }
    ) as unknown as StructuredToolInterface;
    const hookInputs: PreToolUseHookInput[] = [];
    const node = new ToolNode({
      tools: [searchWiki],
      toolRegistry: registry,
      toolDispatch: DISPATCH,
      hookRegistry: capturePreToolUse(hookInputs),
    });

    const result = await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [dispatcherCall('search_wiki', '{"query":"gears"}')],
        }),
      ],
    });

    expect(executed).toEqual([{ query: 'gears' }]);
    expect(hookInputs[0]).toMatchObject({
      toolName: 'search_wiki',
      toolInput: { query: 'gears' },
    });
    expect(messagesOf(result)[0]).toMatchObject({
      name: TOOL_DISPATCHER_NAME,
      tool_call_id: 'call_d',
      content: 'wiki:gears',
    });
  });
});

describe('ToolNode tool dispatch with human-in-the-loop approval', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('asks for approval of the target and runs the edited target arguments', async () => {
    const captured = captureEvents('ran');
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [createToolPolicyHook({ mode: 'bypass', ask: ['search_*'] })],
    });
    const node = createEventNode({
      hookRegistry: hooks,
      humanInTheLoop: { enabled: true },
    });
    const call = dispatcherCall('search_wiki', '{"query":"secret"}');
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('agent', (state: { messages?: BaseMessage[] }) => ({
        messages:
          state.messages?.some((message) => message._getType() === 'tool') ===
          true
            ? [new AIMessage('done')]
            : [new AIMessage({ content: '', tool_calls: [call] })],
      }))
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END)
      .compile({ checkpointer: new MemorySaver() });
    const config: RunnableConfig = {
      configurable: { thread_id: 'thread-dispatch-hitl' },
    };

    const interrupted = await graph.invoke({ messages: [] }, config);

    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.action_requests).toEqual([
      expect.objectContaining({
        tool_call_id: 'call_d',
        name: 'search_wiki',
        arguments: { query: 'secret' },
      }),
    ]);
    expect(captured.requests).toEqual([]);

    const resumed = (await graph.invoke(
      new Command({
        resume: {
          [interrupted.__interrupt__[0].id!]: [
            { type: 'edit', updatedInput: { query: 'public' } },
          ],
        },
      }),
      {
        ...config,
        configurable: {
          ...config.configurable,
          [TOOL_BATCH_REPLAY_KEY]: getToolBatchReplayState(
            interrupted.__interrupt__[0].value
          ),
        },
      }
    )) as { messages: BaseMessage[] };

    expect(captured.requests).toEqual([
      expect.objectContaining({
        id: 'call_d',
        name: 'search_wiki',
        args: { query: 'public' },
      }),
    ]);
    const toolMessage = resumed.messages.find(
      (message) => message._getType() === 'tool'
    ) as ToolMessage;
    expect(toolMessage).toMatchObject({
      name: TOOL_DISPATCHER_NAME,
      tool_call_id: 'call_d',
      content: 'ran:search_wiki',
    });
  });
});

describe('AgentContext tool dispatch binding', () => {
  function boundNames(agentContext: AgentContext): string[] {
    return (agentContext.getToolsForBinding() ?? []).map((boundTool) =>
      'name' in boundTool ? String(boundTool.name) : ''
    );
  }

  function createContext(toolDispatch?: true): AgentContext {
    return AgentContext.fromConfig({
      agentId: 'agent-x',
      provider: Providers.OPENAI,
      toolDefinitions: DEFINITIONS,
      toolRegistry: registry,
      toolDispatch,
    } as t.AgentInputs);
  }

  it('binds the dispatcher last and never binds deferred tools, even when discovered', () => {
    const agentContext = createContext(true);
    expect(boundNames(agentContext)).toEqual([
      'lookup_order',
      'old_search',
      TOOL_DISPATCHER_NAME,
    ]);
    agentContext.markToolsAsDiscovered(['search_wiki']);
    expect(boundNames(agentContext)).toEqual([
      'lookup_order',
      'old_search',
      TOOL_DISPATCHER_NAME,
    ]);
  });

  it('binds discovered deferred tools as before without the dispatcher', () => {
    const agentContext = createContext();
    agentContext.markToolsAsDiscovered(['search_wiki']);
    expect(boundNames(agentContext)).toEqual([
      'lookup_order',
      'search_wiki',
      'old_search',
    ]);
  });

  it('exposes the dispatcher definition with a string arguments field', () => {
    expect(createToolDispatcherDefinition({ name: 'use_tool' })).toMatchObject({
      name: 'use_tool',
      parameters: {
        properties: { arguments: { type: 'string' } },
        required: ['name'],
      },
    });
  });
});

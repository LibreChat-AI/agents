import { z } from 'zod';
import { Command } from '@langchain/langgraph';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  ToolResultProtection,
  ToolResultProtectionResult,
} from '@/protection/toolResult';
import type * as t from '@/types';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { Constants, GraphEvents, Providers } from '@/common';
import { createRemoteSdkRun } from '@/runtime/client';
import { ChatModelStreamHandler } from '@/stream';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';
import { Run } from '@/run';

const RAW = 'reach alice@example.com';
const SAFE = 'reach [EMAIL_1]';

function approve(content: string): ToolResultProtectionResult {
  return {
    version: 1,
    ok: true,
    value: { content, replacements: 1, categories: [] },
  };
}

function recordingPolicy(overrides: Partial<ToolResultProtection> = {}): {
  policy: ToolResultProtection;
  inspected: string[];
} {
  const inspected: string[] = [];
  return {
    inspected,
    policy: {
      version: 1,
      toolNames: ['lookup'],
      timeoutMs: 5000,
      maxAttemptBytes: 1 << 20,
      maxBufferedBytes: 1 << 22,
      classify: () => 'prose',
      inspect: ({ content }) => {
        inspected.push(content);
        return approve(content.replaceAll('alice@example.com', '[EMAIL_1]'));
      },
      ...overrides,
    },
  };
}

function callsFor(name: string, id = 'call-1'): { messages: BaseMessage[] } {
  return {
    messages: [
      new AIMessage({
        content: '',
        tool_calls: [{ id, name, args: {}, type: 'tool_call' }],
      }),
    ],
  };
}

function lookupTool(run: () => unknown): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: 'lookup',
    description: 'Looks something up.',
    schema: z.object({}),
    func: async () => run(),
  });
}

async function failure(work: Promise<unknown>): Promise<Error> {
  return work.then(
    () => new Error('expected a rejection'),
    (error: Error) => error
  );
}

describe('ToolNode tool result protection: in-process tools', () => {
  it('inspects a selected result before hooks, the reference registry and the message', async () => {
    const { policy, inspected } = recordingPolicy();
    const hookOutputs: unknown[] = [];
    const hooks = new HookRegistry();
    hooks.register('PostToolUse', {
      hooks: [
        async (input): Promise<Record<string, never>> => {
          hookOutputs.push((input as { toolOutput?: unknown }).toolOutput);
          return {};
        },
      ],
    });
    const node = new ToolNode({
      tools: [lookupTool(() => RAW)],
      toolResultProtection: policy,
      toolOutputReferences: { enabled: true },
      hookRegistry: hooks,
    });
    const result = (await node.invoke(callsFor('lookup'), {
      configurable: { run_id: 'direct-success' },
    })) as { messages: ToolMessage[] };

    expect(inspected).toEqual([RAW]);
    expect(result.messages[0]).toMatchObject({
      content: SAFE,
      name: 'lookup',
      tool_call_id: 'call-1',
      status: 'success',
    });
    expect(hookOutputs).toEqual([SAFE]);
    expect(
      node._unsafeGetToolOutputRegistry()!.get('direct-success', 'tool0turn0')
    ).toBe(SAFE);
  });

  it('leaves the result alone when no policy is configured', async () => {
    const node = new ToolNode({ tools: [lookupTool(() => RAW)] });
    const result = (await node.invoke(callsFor('lookup'))) as {
      messages: ToolMessage[];
    };
    expect(result.messages[0].content).toBe(RAW);
  });

  it('inspects a selected tool error before the error handler and the message', async () => {
    const { policy, inspected } = recordingPolicy();
    const handled: string[] = [];
    const node = new ToolNode({
      tools: [
        lookupTool(() => {
          throw new Error(RAW);
        }),
      ],
      toolResultProtection: policy,
      errorHandler: async (data): Promise<void> => {
        handled.push(data.error?.message ?? '');
      },
    });
    const result = (await node.invoke(callsFor('lookup'))) as {
      messages: ToolMessage[];
    };

    expect(inspected).toEqual([RAW]);
    expect(handled).toEqual([SAFE]);
    expect(result.messages[0].status).toBe('error');
    expect(result.messages[0].content).toContain(SAFE);
    expect(result.messages[0].content).not.toContain('alice');
  });

  it.each([
    [
      'an artifact',
      (): unknown => [RAW, { rows: [1] }],
      'content_and_artifact',
    ],
    [
      'a Command',
      (): unknown => new Command({ update: { messages: [] } }),
      'content',
    ],
  ] as const)(
    'ends the batch as unsupported when a selected tool returns %s',
    async (_label, run, responseFormat) => {
      const { policy, inspected } = recordingPolicy();
      const selected = tool(async () => run(), {
        name: 'lookup',
        description: 'Looks something up.',
        schema: z.object({}),
        responseFormat,
      });
      const node = new ToolNode({
        tools: [selected],
        toolResultProtection: policy,
      });
      const error = await failure(node.invoke(callsFor('lookup')));
      expect(error).toBeInstanceOf(ToolResultProtectionError);
      expect(error).toMatchObject({ code: 'unsupported' });
      expect(inspected).toEqual([]);
    }
  );

  it('ends the batch with the inspector decision instead of a tool error message', async () => {
    const { policy } = recordingPolicy({
      inspect: () => ({ version: 1, ok: false, error: { code: 'blocked' } }),
    });
    const node = new ToolNode({
      tools: [lookupTool(() => RAW)],
      toolResultProtection: policy,
    });
    const error = await failure(node.invoke(callsFor('lookup')));
    expect(error).toMatchObject({
      name: 'ToolResultProtectionError',
      code: 'blocked',
    });
    expect(error.message).not.toContain('alice');
  });

  class Row {
    constructor(readonly value: number) {}
  }

  it.each([
    ['a class instance', new Row(1)],
    ['a Map', new Map([['key', 1]])],
    ['a function field', { render: (): number => 1 }],
    ['ten thousand rows', Array.from({ length: 10000 }, (_v, i) => ({ i }))],
  ])(
    'passes an unselected tool artifact holding %s through untouched',
    async (_label, artifact) => {
      const { policy, inspected } = recordingPolicy();
      const other = tool(async () => [RAW, artifact], {
        name: 'other',
        description: 'Not selected.',
        schema: z.object({}),
        responseFormat: 'content_and_artifact',
      });
      const node = new ToolNode({
        tools: [other],
        toolResultProtection: policy,
      });
      const result = (await node.invoke(callsFor('other'))) as {
        messages: ToolMessage[];
      };
      expect(result.messages[0].content).toBe(RAW);
      expect(result.messages[0].artifact).toBe(artifact);
      expect(inspected).toEqual([]);
    }
  );

  it('keeps selected tools out of programmatic tool calling', async () => {
    const { policy } = recordingPolicy();
    const captured: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const programmatic = tool(
      async (_args, config) => {
        captured.push(
          config.toolCall as ToolCall & Partial<t.ProgrammaticCache>
        );
        return 'done';
      },
      {
        name: Constants.PROGRAMMATIC_TOOL_CALLING,
        description: 'Run tools with code',
        schema: z.object({ code: z.string() }),
      }
    );
    const lookup = lookupTool(() => RAW);
    const other = tool(async () => 'other', {
      name: 'other',
      description: 'Not selected.',
      schema: z.object({}),
    });
    const node = new ToolNode({
      tools: [programmatic, lookup, other],
      toolRegistry: new Map([
        ['lookup', { name: 'lookup', allowed_callers: ['code_execution'] }],
        ['other', { name: 'other', allowed_callers: ['code_execution'] }],
      ]),
      toolResultProtection: policy,
    });
    await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'ptc',
              name: Constants.PROGRAMMATIC_TOOL_CALLING,
              args: { code: 'print(1)' },
              type: 'tool_call',
            },
          ],
        }),
      ],
    });

    expect(captured).toHaveLength(1);
    expect([...captured[0].toolMap!.keys()]).toEqual(['other']);
    expect(captured[0].toolDefs?.map((def) => def.name)).toEqual(['other']);
    expect(captured[0].disallowedToolDefs).toEqual([{ name: 'lookup' }]);
  });
});

interface HostRun {
  run: Run<t.IState>;
  wireInputs: string[];
  completions: string[];
  executions: () => number;
}

async function createHostRun(args: {
  id: string;
  policy?: ToolResultProtection;
  toolName?: string;
  result: (id: string) => t.ToolExecuteResult;
}): Promise<HostRun> {
  const wireInputs: string[] = [];
  const completions: string[] = [];
  const toolName = args.toolName ?? 'lookup';
  let executions = 0;
  class Model extends FakeChatModel {
    override async *_streamResponseChunks(
      ...params: Parameters<FakeChatModel['_streamResponseChunks']>
    ): ReturnType<FakeChatModel['_streamResponseChunks']> {
      wireInputs.push(JSON.stringify(params[0]));
      yield* super._streamResponseChunks(...params);
    }
  }
  const run = await Run.create<t.IState>({
    runId: args.id,
    graphConfig: {
      type: 'standard',
      llmConfig: { provider: Providers.OPENAI },
      instructions: 'Use the tool.',
      toolDefinitions: [
        {
          name: toolName,
          description: 'Host executed.',
          parameters: { type: 'object', properties: {} },
        },
      ],
    },
    customHandlers: {
      [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, data): void => {
          const request = data as t.ToolExecuteBatchRequest;
          executions++;
          const result = args.result(request.toolCalls[0].id);
          request.onResult?.(result);
          request.resolve([result]);
        },
      },
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (_event, data): void => {
          completions.push(JSON.stringify(data));
        },
      },
    },
    toolResultProtection: args.policy,
    returnContent: true,
    skipCleanup: true,
  });
  run.Graph!.overrideModel = new Model({
    responses: ['', 'All done'],
    toolCalls: [{ id: 'call-1', name: toolName, args: {}, type: 'tool_call' }],
  });
  return { run, wireInputs, completions, executions: () => executions };
}

function start(host: HostRun, id: string): Promise<unknown> {
  return host.run.processStream(
    { messages: [new HumanMessage('Look it up')] },
    { version: 'v2', configurable: { thread_id: id } }
  );
}

describe('Run tool result protection: host-executed tools', () => {
  it('publishes and reuses only the inspected result', async () => {
    const { policy, inspected } = recordingPolicy();
    const host = await createHostRun({
      id: 'host-success',
      policy,
      result: (toolCallId) => ({ toolCallId, status: 'success', content: RAW }),
    });
    await start(host, 'host-success');

    expect(host.executions()).toBe(1);
    expect(inspected).toEqual([RAW]);
    expect(host.wireInputs).toHaveLength(2);
    expect(host.wireInputs[1]).toContain(SAFE);
    expect(host.wireInputs[1]).not.toContain('alice');
    expect(host.completions.join('')).toContain(SAFE);
    expect(host.completions.join('')).not.toContain('alice');
    expect(JSON.stringify(host.run.getRunMessages())).not.toContain('alice');
  });

  it('sends the raw result when no policy is configured', async () => {
    const host = await createHostRun({
      id: 'host-off',
      result: (toolCallId) => ({ toolCallId, status: 'success', content: RAW }),
    });
    await start(host, 'host-off');
    expect(host.wireInputs[1]).toContain(RAW);
    expect(host.completions.join('')).toContain(RAW);
  });

  it('inspects a host error message', async () => {
    const { policy, inspected } = recordingPolicy();
    const host = await createHostRun({
      id: 'host-error',
      policy,
      result: (toolCallId) => ({
        toolCallId,
        status: 'error',
        content: RAW,
        errorMessage: RAW,
      }),
    });
    await start(host, 'host-error');

    expect(inspected).toEqual([RAW]);
    expect(host.wireInputs[1]).toContain(SAFE);
    expect(host.wireInputs[1]).not.toContain('alice');
    expect(host.completions.join('')).not.toContain('alice');
  });

  const hostFailures: Array<
    [
      string,
      string,
      Pick<t.ToolExecuteResult, 'content' | 'artifact'>,
      Partial<ToolResultProtection>,
    ]
  > = [
    [
      'the inspector blocks it',
      'blocked',
      { content: RAW },
      {
        inspect: (): ToolResultProtectionResult => ({
          version: 1,
          ok: false,
          error: { code: 'blocked' },
        }),
      },
    ],
    [
      'it carries an artifact',
      'unsupported',
      { content: RAW, artifact: {} },
      {},
    ],
    [
      'its content is structured',
      'unsupported',
      { content: [{ type: 'text', text: RAW }] },
      {},
    ],
  ];

  it.each(hostFailures)(
    'ends the run without another model call when %s',
    async (_label, code, fields, overrides) => {
      const { policy } = recordingPolicy(overrides);
      const id = `host-fail-${code}-${_label.length}`;
      const host = await createHostRun({
        id,
        policy,
        result: (toolCallId) => ({ toolCallId, status: 'success', ...fields }),
      });
      const error = await failure(start(host, id));

      expect(error).toMatchObject({ name: 'ToolResultProtectionError', code });
      expect(host.wireInputs).toHaveLength(1);
      expect(host.completions.join('')).not.toContain('alice');
    }
  );

  it('leaves an unselected host tool alone, artifact and structured content included', async () => {
    const { policy, inspected } = recordingPolicy();
    const host = await createHostRun({
      id: 'host-unselected',
      policy,
      toolName: 'other',
      result: (toolCallId) => ({
        toolCallId,
        status: 'success',
        content: [{ type: 'text', text: RAW }],
        artifact: { rows: Array.from({ length: 10000 }, (_v, i) => ({ i })) },
      }),
    });
    await start(host, 'host-unselected');

    expect(inspected).toEqual([]);
    expect(host.wireInputs).toHaveLength(2);
    expect(host.wireInputs[1]).toContain(RAW);
  });
});

describe('Run tool result protection: across turns and graphs', () => {
  async function turn(args: {
    id: string;
    policy: ToolResultProtection;
    messages: BaseMessage[];
    callsTool: boolean;
  }): Promise<BaseMessage[]> {
    const run = await Run.create<t.IState>({
      runId: args.id,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
        tools: [lookupTool(() => RAW)],
      },
      toolResultProtection: args.policy,
      skipCleanup: true,
    });
    run.Graph!.overrideModel = new FakeChatModel({
      responses: args.callsTool ? ['', 'First answer'] : ['Second answer'],
      toolCalls: args.callsTool
        ? [{ id: 'call-1', name: 'lookup', args: {}, type: 'tool_call' }]
        : [],
    });
    await run.processStream(
      { messages: args.messages },
      { version: 'v2', configurable: { thread_id: args.id } }
    );
    return run.getRunMessages() ?? [];
  }

  it('accepts a follow-up turn whose history the host rebuilt from storage', async () => {
    const { policy, inspected } = recordingPolicy();
    const question = new HumanMessage('Look it up');
    const first = await turn({
      id: 'turn-1',
      policy,
      messages: [question],
      callsTool: true,
    });
    expect(first.map((message) => message.getType())).toEqual([
      'ai',
      'tool',
      'ai',
    ]);
    expect(first[1].content).toBe(SAFE);

    const stored: BaseMessage[] = [
      new HumanMessage('Look it up'),
      new AIMessage({
        content: '',
        tool_calls: (first[0] as AIMessage).tool_calls,
      }),
      new ToolMessage({
        content: first[1].content as string,
        name: 'lookup',
        tool_call_id: 'call-1',
      }),
      new AIMessage({ content: first[2].content as string }),
      new HumanMessage('And again?'),
    ];
    const second = await turn({
      id: 'turn-2',
      policy,
      messages: stored,
      callsTool: false,
    });

    expect(second.at(-1)?.content).toBe('Second answer');
    expect(inspected).toEqual([RAW]);
  });

  it('inspects a selected tool run by a subagent', async () => {
    const { policy, inspected } = recordingPolicy();
    let executions = 0;
    const childInputs: string[] = [];
    class ChildModel extends FakeChatModel {
      override async *_streamResponseChunks(
        ...params: Parameters<FakeChatModel['_streamResponseChunks']>
      ): ReturnType<FakeChatModel['_streamResponseChunks']> {
        childInputs.push(JSON.stringify(params[0]));
        yield* super._streamResponseChunks(...params);
      }
    }
    const run = await Run.create<t.IState>({
      runId: 'child-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'parent',
            provider: Providers.OPENAI,
            instructions: 'Delegate.',
            subagentConfigs: [
              {
                type: 'worker',
                name: 'Worker',
                description: 'Looks things up.',
                agentInputs: {
                  agentId: 'child',
                  provider: Providers.OPENAI,
                  instructions: 'Use the tool.',
                  tools: [
                    lookupTool(() => {
                      executions++;
                      return RAW;
                    }),
                  ],
                },
              },
            ],
          },
        ],
      },
      toolResultProtection: policy,
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['', 'Parent answer'],
      toolCalls: [
        {
          id: 'delegate-1',
          name: Constants.SUBAGENT,
          args: { description: 'Look it up.', subagent_type: 'worker' },
          type: 'tool_call',
        },
      ],
    });
    run.Graph!.setSubagentModelOverride(
      new ChildModel({
        responses: ['', 'Child answer'],
        toolCalls: [
          { id: 'call-1', name: 'lookup', args: {}, type: 'tool_call' },
        ],
      })
    );
    await run.processStream(
      { messages: [new HumanMessage('Look it up')] },
      { version: 'v2', configurable: { thread_id: 'child-run' } }
    );

    expect(executions).toBe(1);
    expect(inspected).toEqual([RAW]);
    expect(childInputs).toHaveLength(2);
    expect(childInputs[1]).toContain(SAFE);
    expect(childInputs[1]).not.toContain('alice');
  });
});

describe('Run tool result protection: configuration', () => {
  const graphConfig = {
    type: 'standard' as const,
    llmConfig: { provider: Providers.OPENAI },
    tools: [lookupTool(() => RAW)],
  };

  it('rejects an invalid policy when the run is created', async () => {
    const { policy } = recordingPolicy({ toolNames: [] });
    const error = await failure(
      Run.create({
        runId: 'invalid',
        graphConfig,
        toolResultProtection: policy,
      })
    );
    expect(error).toMatchObject({
      name: 'ToolResultProtectionError',
      code: 'incompatible',
    });
  });

  it('rejects a policy placed on the graph config', async () => {
    const { policy } = recordingPolicy();
    const error = await failure(
      Run.create({
        runId: 'misplaced',
        graphConfig: {
          ...graphConfig,
          toolResultProtection: policy,
        } as t.RunConfig['graphConfig'],
      })
    );
    expect(error).toMatchObject({
      name: 'ToolResultProtectionError',
      code: 'incompatible',
    });
  });

  it('rejects a selected tool that is also client delegated', async () => {
    const { policy } = recordingPolicy();
    const error = await failure(
      Run.create({
        runId: 'delegated',
        graphConfig,
        toolResultProtection: policy,
        clientDelegatedToolNames: ['lookup'],
        customHandlers: {
          [GraphEvents.ON_MODEL_RESPONSE]: { handle: (): void => {} },
        },
      })
    );
    expect(error).toMatchObject({
      name: 'ToolResultProtectionError',
      code: 'unsupported',
    });
  });

  it('rejects a remote SDK run, which cannot carry the inspector', async () => {
    const { policy } = recordingPolicy();
    const error = await failure(
      createRemoteSdkRun({
        config: { runId: 'remote', graphConfig, toolResultProtection: policy },
        identity: {} as Parameters<typeof createRemoteSdkRun>[0]['identity'],
        remote: {} as Parameters<typeof createRemoteSdkRun>[0]['remote'],
      })
    );
    expect(error).toMatchObject({
      name: 'ToolResultProtectionError',
      code: 'unsupported',
    });
  });
});

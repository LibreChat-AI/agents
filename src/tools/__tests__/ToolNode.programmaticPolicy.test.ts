import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { ToolNode } from '../ToolNode';
import * as events from '@/utils/events';
import { Constants, GraphEvents } from '@/common';

describe('ToolNode programmatic caller policy', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each([false, true])('enforces direct policy before executing or dispatching a mixed batch (event=%s)', async (eventDrivenMode) => {
    const called: string[] = [];
    const definitions: t.LCTool[] = [
      { name: 'code_only', allowed_callers: ['code_execution'], defer_loading: true },
      { name: 'undiscovered', allowed_callers: ['direct'], defer_loading: true },
      { name: 'disabled', allowed_callers: [] },
      { name: 'allowed', allowed_callers: ['direct'], defer_loading: true },
    ];
    const tools = definitions.map((definition) => tool(async () => { called.push(definition.name); return 'ok'; }, { name: definition.name, schema: z.object({}) }));
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(async (event, data): Promise<void> => {
      if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
      const batch = data as t.ToolExecuteBatchRequest;
      called.push(...batch.toolCalls.map((call) => call.name));
      batch.resolve(batch.toolCalls.map((call) => ({ toolCallId: call.id, status: 'success', content: 'ok' })));
    });
    const node = new ToolNode({ tools, eventDrivenMode, toolRegistry: new Map(definitions.map((definition) => [definition.name, definition])),
      toolCallStepIds: new Map(definitions.map(({ name }) => [name, `step-${name}`])),
      toolDefinitions: eventDrivenMode ? new Map(definitions.map(({ name }) => [name, { name }])) : undefined,
      getDiscoveredToolNames: () => ['code_only', 'allowed'],
    });
    const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: definitions.map(({ name }) => ({ id: name, name, args: {} })) })] }) as ToolMessage[] | { messages: ToolMessage[] };
    const messages = Array.isArray(result) ? result : result.messages;
    expect(called).toEqual(['allowed']);
    expect(messages).toHaveLength(4);
    expect(messages.slice(0, 3).map((message) => 'status' in message ? message.status : undefined)).toEqual(['error', 'error', 'error']);
    expect(messages.map((message) => 'tool_call_id' in message ? message.tool_call_id : undefined)).toEqual(definitions.map(({ name }) => name));
  });

  it('separates programmatic tools from direct-only preflight definitions', async () => {
    const capturedConfigs: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const ptcTool = tool(
      async (_args, config) => {
        capturedConfigs.push(
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
    const programmaticTool = tool(async () => 'programmatic', {
      name: 'programmatic_tool',
      description: 'Programmatic tool',
      schema: z.object({}),
    });
    const directTool = tool(async () => 'direct', {
      name: 'direct_tool',
      description: 'Direct tool',
      schema: z.object({}),
    });
    const toolRegistry: t.LCToolRegistry = new Map([
      [
        programmaticTool.name,
        {
          name: programmaticTool.name,
          allowed_callers: ['code_execution'],
        },
      ],
      [directTool.name, { name: directTool.name, allowed_callers: ['direct'] }],
    ]);
    const node = new ToolNode({
      tools: [ptcTool, programmaticTool, directTool],
      toolRegistry,
    });

    await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'ptc-call',
              name: Constants.PROGRAMMATIC_TOOL_CALLING,
              args: { code: 'print("done")' },
            },
          ],
        }),
      ],
    });

    expect(capturedConfigs).toHaveLength(1);
    expect(capturedConfigs[0].toolDefs).toEqual([
      { name: 'programmatic_tool', allowed_callers: ['code_execution'] },
    ]);
    expect(capturedConfigs[0].disallowedToolDefs).toEqual([
      { name: 'direct_tool' },
    ]);
    expect(capturedConfigs[0].toolMap).toEqual(
      new Map([['programmatic_tool', programmaticTool]])
    );
    expect(capturedConfigs[0].programmaticToolName).toBe(
      Constants.PROGRAMMATIC_TOOL_CALLING
    );
  });

  it('projects deferred policy definitions only after discovery', async () => {
    const capturedConfigs: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const discovered = new Set<string>();
    const ptcTool = tool(
      async (_args, config) => {
        capturedConfigs.push(
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
    const directTool = tool(async () => 'direct', {
      name: 'deferred_direct_tool',
      description: 'Deferred direct tool',
      schema: z.object({}),
    });
    const node = new ToolNode({
      tools: [ptcTool, directTool],
      toolRegistry: new Map([
        [
          directTool.name,
          {
            name: directTool.name,
            allowed_callers: ['direct'],
            defer_loading: true,
          },
        ],
      ]),
      getDiscoveredToolNames: () => [...discovered],
    });
    const invoke = async (id: string): Promise<void> => {
      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id,
                name: Constants.PROGRAMMATIC_TOOL_CALLING,
                args: { code: 'print("done")' },
              },
            ],
          }),
        ],
      });
    };

    await invoke('before-discovery');
    discovered.add(directTool.name);
    await invoke('after-discovery');

    expect(capturedConfigs[0].disallowedToolDefs).toEqual([]);
    expect(capturedConfigs[1].disallowedToolDefs).toEqual([
      { name: directTool.name },
    ]);
  });

  it('dispatches the same live caller projection in event-driven mode', async () => {
    const discovered = new Set<string>();
    const snapshots: t.CallerCapabilityProjectionSnapshot[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        snapshots.push(batch.callerCapabilityProjection!);
        batch.resolve(
          batch.toolCalls.map((toolCall) => ({
            toolCallId: toolCall.id,
            status: 'success',
            content: 'done',
          }))
        );
      });

    const eventTool = tool(async () => 'should dispatch', {
      name: 'event_tool',
      description: 'Event tool',
      schema: z.object({}),
    });
    const toolDefinitions: t.LCToolRegistry = new Map([
      [eventTool.name, { name: eventTool.name }],
      [
        'deferred_programmatic_tool',
        {
          name: 'deferred_programmatic_tool',
          allowed_callers: ['code_execution'],
          defer_loading: true,
        },
      ],
    ]);
    const toolRegistry: t.LCToolRegistry = new Map([
      [
        'deferred_direct_tool',
        {
          name: 'deferred_direct_tool',
          allowed_callers: ['direct'],
          defer_loading: true,
        },
      ],
    ]);
    const node = new ToolNode({
      tools: [eventTool],
      toolRegistry,
      toolDefinitions,
      eventDrivenMode: true,
      getDiscoveredToolNames: () => [...discovered],
      toolCallStepIds: new Map([
        ['before-discovery', 'step-before'],
        ['after-discovery', 'step-after'],
      ]),
    });
    const invoke = async (id: string): Promise<void> => {
      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id, name: eventTool.name, args: {} }],
          }),
        ],
      });
    };

    await invoke('before-discovery');
    discovered.add('deferred_programmatic_tool');
    discovered.add('deferred_direct_tool');
    await invoke('after-discovery');

    expect(snapshots).toEqual([
      {
        version: 1,
        directToolNames: ['event_tool'],
        codeExecutionToolNames: [],
        directOnlyToolNames: ['event_tool'],
        codeExecutionOnlyToolNames: [],
      },
      {
        version: 1,
        directToolNames: ['event_tool', 'deferred_direct_tool'],
        codeExecutionToolNames: ['deferred_programmatic_tool'],
        directOnlyToolNames: ['event_tool', 'deferred_direct_tool'],
        codeExecutionOnlyToolNames: ['deferred_programmatic_tool'],
      },
    ]);
  });

  it('keeps schema-only event definitions out of a direct runner cache', async () => {
    const capturedConfigs: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const ptcTool = tool(
      async (_args, config) => {
        capturedConfigs.push(
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
    const eventStub = tool(async () => 'schema stub should not run', {
      name: 'event_programmatic_tool',
      description: 'Event programmatic tool',
      schema: z.object({}),
    });
    const node = new ToolNode({
      tools: [ptcTool, eventStub],
      eventDrivenMode: true,
      directToolNames: new Set([Constants.PROGRAMMATIC_TOOL_CALLING]),
      toolRegistry: new Map([
        [
          eventStub.name,
          {
            name: eventStub.name,
            allowed_callers: ['code_execution'],
          },
        ],
      ]),
      toolDefinitions: new Map([
        [
          eventStub.name,
          {
            name: eventStub.name,
            allowed_callers: ['code_execution'],
          },
        ],
      ]),
    });

    await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'local-ptc',
              name: Constants.PROGRAMMATIC_TOOL_CALLING,
              args: { code: 'event_programmatic_tool "{}"' },
            },
          ],
        }),
      ],
    });

    expect(capturedConfigs).toHaveLength(1);
    expect(capturedConfigs[0].toolDefs).toEqual([]);
    expect(capturedConfigs[0].toolMap).toEqual(new Map());
  });

  it('keeps synthesized code-only implementations in the runner cache', () => {
    const node = new ToolNode({
      tools: [],
      eventDrivenMode: true,
      toolExecution: { engine: 'local' },
      toolRegistry: new Map([
        [
          Constants.WRITE_FILE,
          {
            name: Constants.WRITE_FILE,
            allowed_callers: ['code_execution'],
          },
        ],
      ]),
    });

    const cache = (
      node as unknown as {
        getProgrammaticTools(): t.ProgrammaticCache;
      }
    ).getProgrammaticTools();

    expect(cache.toolDefs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: Constants.WRITE_FILE,
          allowed_callers: ['code_execution'],
        }),
      ])
    );
    expect(cache.toolMap.get(Constants.WRITE_FILE)?.name).toBe(
      Constants.WRITE_FILE
    );
  });

  it('does not treat unsupported Cloudflare allowlist names as implementations', () => {
    const unsupportedName = 'unsupported_cloudflare_tool';
    const unsupportedStub = tool(async () => 'schema stub should not run', {
      name: unsupportedName,
      description: 'Unsupported Cloudflare tool',
      schema: z.object({}),
    });
    const node = new ToolNode({
      tools: [unsupportedStub],
      eventDrivenMode: true,
      toolExecution: {
        engine: 'cloudflare-sandbox',
        cloudflare: {
          codingToolNames: [
            Constants.PROGRAMMATIC_TOOL_CALLING,
            unsupportedName,
          ],
          sandbox: {
            exec: async () => ({ exitCode: 0, stdout: '', stderr: '' }),
            readFile: async () => '',
            writeFile: async () => undefined,
            mkdir: async () => undefined,
            listFiles: async () => [],
            deleteFile: async () => undefined,
          },
        },
      },
      toolRegistry: new Map([
        [
          unsupportedName,
          {
            name: unsupportedName,
            allowed_callers: ['code_execution'],
          },
        ],
      ]),
      toolDefinitions: new Map([
        [
          unsupportedName,
          {
            name: unsupportedName,
            allowed_callers: ['code_execution'],
          },
        ],
      ]),
    });

    const cache = (
      node as unknown as {
        getProgrammaticTools(): t.ProgrammaticCache;
      }
    ).getProgrammaticTools();

    expect(cache.toolDefs).toEqual([]);
    expect(cache.toolMap).toEqual(new Map());
  });
});

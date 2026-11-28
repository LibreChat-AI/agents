import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { ToolNode } from '@/tools/ToolNode';
import { Constants, GraphEvents } from '@/common';

describe('ToolNode — defer read_file after code-writing tools in the same batch', () => {
  it('runs execute_code before read_file in direct mode while preserving output order', async () => {
    const events: string[] = [];
    const writtenFiles = new Set<string>();

    const executeCodeTool = tool(
      async ({ filename }: { filename: string }) => {
        events.push(`execute_code:start:${filename}`);
        await new Promise((resolve) => setTimeout(resolve, 30));
        writtenFiles.add(filename);
        events.push(`execute_code:end:${filename}`);
        return `wrote ${filename}`;
      },
      {
        name: Constants.EXECUTE_CODE,
        description: 'Execute code',
        schema: z.object({ filename: z.string() }),
      }
    );

    const readFileTool = tool(
      async ({ path }: { path: string }) => {
        events.push(`read_file:start:${path}`);
        if (!writtenFiles.has(path)) {
          throw new Error(`No such file or directory: ${path}`);
        }
        events.push(`read_file:end:${path}`);
        return `contents of ${path}`;
      },
      {
        name: 'read_file',
        description: 'Read a file',
        schema: z.object({ path: z.string() }),
      }
    );

    const node = new ToolNode({
      tools: [executeCodeTool, readFileTool],
      handleToolErrors: false,
    });

    // Place read_file both before and after execute_code to verify order preservation
    const aiMessage = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'call_read_1',
          name: 'read_file',
          args: { path: '/mnt/data/one.png' },
        },
        {
          id: 'call_exec',
          name: Constants.EXECUTE_CODE,
          args: { filename: '/mnt/data/one.png' },
        },
        {
          id: 'call_read_2',
          name: 'read_file',
          args: { path: '/mnt/data/one.png' },
        },
      ] as ToolCall[],
    });

    const result = (await node.invoke({ messages: [aiMessage] })) as {
      messages: ToolMessage[];
    };

    expect(events.indexOf('execute_code:end:/mnt/data/one.png')).toBeLessThan(
      events.indexOf('read_file:start:/mnt/data/one.png')
    );
    expect(result.messages.map((m) => m.tool_call_id)).toEqual([
      'call_read_1',
      'call_exec',
      'call_read_2',
    ]);
    expect(result.messages.map((m) => m.content)).toEqual([
      'contents of /mnt/data/one.png',
      'wrote /mnt/data/one.png',
      'contents of /mnt/data/one.png',
    ]);
  });

  it('runs bash_tool before multiple read_file calls and keeps read_file calls parallel', async () => {
    const events: string[] = [];
    let activeReads = 0;
    let maxConcurrentReads = 0;

    const bashTool = tool(
      async () => {
        events.push('bash:start');
        await new Promise((resolve) => setTimeout(resolve, 25));
        events.push('bash:end');
        return 'ok';
      },
      {
        name: 'bash_tool',
        description: 'Run bash',
        schema: z.object({}),
      }
    );

    const readFileTool = tool(
      async ({ path }: { path: string }) => {
        events.push(`read:start:${path}`);
        activeReads++;
        maxConcurrentReads = Math.max(maxConcurrentReads, activeReads);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeReads--;
        events.push(`read:end:${path}`);
        return `read:${path}`;
      },
      {
        name: 'read_file',
        description: 'Read a file',
        schema: z.object({ path: z.string() }),
      }
    );

    const node = new ToolNode({
      tools: [bashTool, readFileTool],
      handleToolErrors: false,
    });

    const aiMessage = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'call_bash', name: 'bash_tool', args: {} },
        { id: 'call_r1', name: 'read_file', args: { path: 'a.png' } },
        { id: 'call_r2', name: 'read_file', args: { path: 'b.png' } },
      ] as ToolCall[],
    });

    const result = (await node.invoke({ messages: [aiMessage] })) as {
      messages: ToolMessage[];
    };

    const bashEnd = events.indexOf('bash:end');
    expect(bashEnd).toBeGreaterThan(-1);
    expect(bashEnd).toBeLessThan(events.indexOf('read:start:a.png'));
    expect(bashEnd).toBeLessThan(events.indexOf('read:start:b.png'));
    expect(maxConcurrentReads).toBe(2);
    expect(result.messages.map((m) => m.tool_call_id)).toEqual([
      'call_bash',
      'call_r1',
      'call_r2',
    ]);
  });

  it('defers read_file after execute_code in event-driven ON_TOOL_EXECUTE mode', async () => {
    const dispatchedBatches: string[][] = [];
    const writtenFiles = new Set<string>();

    const toolDefinitions: t.LCTool[] = [
      {
        name: Constants.EXECUTE_CODE,
        description: 'Execute code',
        parameters: {
          type: 'object',
          properties: { filename: { type: 'string' } },
        },
      },
      {
        name: 'read_file',
        description: 'Read file',
        parameters: {
          type: 'object',
          properties: { path: { type: 'string' } },
        },
      },
    ];

    const node = new ToolNode({
      toolDefinitions: new Map(toolDefinitions.map((d) => [d.name, d])),
      toolMap: new Map(),
      eventDrivenMode: true,
    });

    const dispatchSpy = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (eventName, data): Promise<void> => {
        if (eventName !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        dispatchedBatches.push(batch.toolCalls.map((c) => c.name));
        const results: t.ToolExecuteResult[] = [];
        for (const call of batch.toolCalls) {
          if (call.name === Constants.EXECUTE_CODE) {
            await new Promise((resolve) => setTimeout(resolve, 20));
            writtenFiles.add(String(call.args.filename));
            results.push({
              toolCallId: call.id,
              content: `wrote ${String(call.args.filename)}`,
              status: 'success',
            });
          } else if (call.name === 'read_file') {
            const path = String(call.args.path);
            results.push({
              toolCallId: call.id,
              content: writtenFiles.has(path)
                ? `contents of ${path}`
                : `missing ${path}`,
              status: writtenFiles.has(path) ? 'success' : 'error',
            });
          }
        }
        batch.resolve(results);
      });

    try {
      const aiMessage = new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_read',
            name: 'read_file',
            args: { path: '/mnt/data/chart.png' },
          },
          {
            id: 'call_exec',
            name: Constants.EXECUTE_CODE,
            args: { filename: '/mnt/data/chart.png' },
          },
        ] as ToolCall[],
      });

      const result = (await node.invoke({ messages: [aiMessage] })) as {
        messages: ToolMessage[];
      };

      expect(dispatchedBatches).toEqual([
        [Constants.EXECUTE_CODE],
        ['read_file'],
      ]);
      expect(result.messages.map((m) => m.tool_call_id)).toEqual([
        'call_read',
        'call_exec',
      ]);
      expect(result.messages.map((m) => m.content)).toEqual([
        'contents of /mnt/data/chart.png',
        'wrote /mnt/data/chart.png',
      ]);
    } finally {
      dispatchSpy.mockRestore();
    }
  });
});

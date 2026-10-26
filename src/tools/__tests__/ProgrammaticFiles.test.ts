import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import type { CodeEnvFile } from '@/types';
import {
  executeTools,
  requireContinuationFiles,
  programmaticWireResults,
  retainProgrammaticAttachments,
} from '../ProgrammaticToolCalling';
import { Constants } from '@/common';
const file: CodeEnvFile = {
  id: 'synthetic-file',
  name: 'fixture.pdf',
  storage_session_id: 'synthetic-storage',
  resource_id: 'synthetic-user',
  kind: 'user',
};
test('PTC preserves owned MCP references and trusted execution context separately from text', async () => {
  let context: string | undefined;
  const tool = new DynamicStructuredTool({
    name: 'download_mcp_fixture',
    description: 'Synthetic file',
    schema: z.object({}),
    responseFormat: 'content_and_artifact',
    func: async (_input, _runManager, config) => {
      const value = config?.metadata?.executionContext;
      context = typeof value === 'string' ? value : undefined;
      return ['ready', { files: [file] }];
    },
  });
  const mcpTool = Object.assign(tool, { mcp: true });
  const result = await executeTools(
    [{ id: 'call_001', name: tool.name, input: {} }],
    new Map([[tool.name, mcpTool]]),
    Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
    { metadata: { executionContext: 'synthetic-context' } }
  );
  expect(result[0].result).toBe('ready');
  expect(result[0].files).toEqual([file]);
  expect(context).toBe('synthetic-context');
  expect(() =>
    requireContinuationFiles(result, { status: 'tool_call_required' })
  ).toThrow('unsupported');
  expect(() =>
    requireContinuationFiles(result, {
      status: 'tool_call_required',
      continuation_files: 1,
    })
  ).not.toThrow();
});

test('host metadata stays local and is retained on the outer artifact', () => {
  const results = [
    {
      call_id: 'call_001',
      result: 'ready',
      is_error: false,
      files: [file],
      mcp_files: [{ file_id: 'synthetic-file' }],
    },
  ];
  expect(programmaticWireResults(results)[0]).not.toHaveProperty('mcp_files');
  expect(programmaticWireResults(results)[0].files).toEqual([file]);
  expect(
    retainProgrammaticAttachments(['done', {}], results)[1].mcp_files
  ).toEqual([{ file_id: 'synthetic-file' }]);
});

test('passes cancellation to the nested tool without changing ordinary tool inputs', async () => {
  const controller = new AbortController();
  const ordinary = new DynamicStructuredTool({
    name: 'ordinary',
    description: 'Synthetic ordinary tool',
    schema: z.object({ value: z.string() }),
    func: async (input, _manager, config) => {
      expect(config?.signal).toBe(controller.signal);
      return input.value;
    },
  });
  const results = await executeTools(
    [{ id: 'call_001', name: ordinary.name, input: { value: 'synthetic' } }],
    new Map([[ordinary.name, ordinary]]),
    Constants.PROGRAMMATIC_TOOL_CALLING,
    { signal: controller.signal }
  );
  expect(results[0].result).toBe('synthetic');
  expect(results[0].files).toBeUndefined();
});

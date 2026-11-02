import { tool } from '@langchain/core/tools';
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, writeFile, rm } from 'fs/promises';
import { createLocalReadFileTool, createLocalCodingToolDefinitions } from '../local/LocalCodingTools';
import { createLocalProgrammaticProfile } from '../local/LocalProgrammaticProfile';
import { resolveLocalToolsForBinding, resolveLocalToolRegistry } from '../local/resolveLocalExecutionTools';
import { createToolSearch } from '../ToolSearch';
import type * as t from '@/types';
import { Constants } from '@/common';

const parameters: t.JsonSchemaType = {
  type: 'object',
  properties: {
    payload: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'], additionalProperties: false },
  },
  required: ['payload'],
};

it('returns complete canonical schemas without changing legacy discovery output', async () => {
  const registry: t.LCToolRegistry = new Map([['fetch_mcp_example', {
    name: 'fetch_mcp_example', description: 'Full fetch description', parameters,
    defer_loading: true, allowed_callers: ['code_execution'],
  }]]);
  for (const includeSchemas of [false, true]) {
    const search = createToolSearch({ mode: 'local', toolRegistry: registry, includeSchemas, mcpNameFormat: 'base' });
    const result = JSON.parse(String(await search.invoke({ query: 'fetch' }))) as { tools: t.LCTool[] };
    expect(result.tools).toHaveLength(1);
    expect(result.tools[0].name).toBe(includeSchemas ? 'fetch_mcp_example' : 'fetch');
    expect(result.tools[0].parameters).toEqual(includeSchemas ? parameters : undefined);
    const empty = String(await search.invoke({ query: 'notfound' }));
    expect(empty).not.toContain('Full fetch description');
  }
});

it('does not broaden explicit host caller restrictions or mutate supplied definitions', () => {
  const bound = tool(async () => 'ok', { name: 'fetch', schema: parameters });
  const definition: t.LCTool = { name: 'fetch', parameters, allowed_callers: ['direct'] };
  const registry = new Map([['fetch', definition]]);
  expect(() => createLocalProgrammaticProfile({ tools: [bound], toolRegistry: registry })).toThrow('does not permit code execution');
  delete definition.allowed_callers;
  expect(() => createLocalProgrammaticProfile({ tools: [bound], toolRegistry: registry })).toThrow('does not permit code execution');
  definition.allowed_callers = ['direct', 'code_execution'];
  const profile = createLocalProgrammaticProfile({ tools: [bound], toolRegistry: registry });
  expect(definition.allowed_callers).toEqual(['direct', 'code_execution']);
  expect(definition.defer_loading).toBeUndefined();
  expect(profile.toolRegistry.get('fetch')?.allowed_callers).toEqual(['code_execution']);
});

it('does not override a selected local implementation caller restriction', () => {
  const toolRegistry: t.LCToolRegistry = new Map([['read_file', {
    name: 'read_file', allowed_callers: ['direct'],
  }]]);
  expect(() => createLocalProgrammaticProfile({
    local: { codingToolNames: ['read_file'] }, toolRegistry,
  })).toThrow('does not permit code_execution');
});

it('rejects unknown local names instead of silently dropping an implementation', () => {
  expect(() => createLocalProgrammaticProfile({
    local: { codingToolNames: ['read_flie'] },
  })).toThrow('Unsupported profile local tool');
});

it('reserves unselected built-in names against connector replacement', () => {
  const bound = tool(async () => 'host implementation', { name: 'read_file', schema: parameters });
  expect(() => createLocalProgrammaticProfile({
    local: { codingToolNames: [] }, tools: [bound],
    toolRegistry: new Map([['read_file', {
      name: 'read_file', parameters, allowed_callers: ['code_execution'],
    }]]),
  })).toThrow('Reserved profile tool');
});

it.each(['compact', 'hybrid'] as const)('binds only selected local tools in %s mode and keeps connector schemas deferred', (mode) => {
  const bound = tool(async () => 'ok', { name: 'fetch', schema: parameters });
  const profile = createLocalProgrammaticProfile({
    mode,
    local: { codingToolNames: ['read_file', 'bash_tool', 'read_file'] },
    tools: [bound],
    toolRegistry: new Map([['fetch', { name: 'fetch', parameters, allowed_callers: ['direct', 'code_execution'] }]]),
  });
  const toolRegistry = resolveLocalToolRegistry(profile);
  for (const discoveredToolNames of [new Set<string>(), new Set(['fetch'])]) {
    const boundTools = resolveLocalToolsForBinding({ ...profile, toolRegistry, discoveredToolNames });
    expect(boundTools?.map((entry) => 'name' in entry ? entry.name : undefined).sort()).toEqual(
      (mode === 'hybrid' ? ['read_file', 'bash_tool', 'tool_search', 'run_tools_with_code'] : ['tool_search', 'run_tools_with_code']).sort()
    );
  }
  expect(profile.toolExecution.local?.codingToolNames).toEqual(['read_file', 'bash_tool', Constants.PROGRAMMATIC_TOOL_CALLING]);
  expect(toolRegistry?.has('write_file')).toBe(false);
  expect(toolRegistry?.get('fetch')).toMatchObject({ allowed_callers: ['code_execution'], defer_loading: true });
  expect(toolRegistry?.get('read_file')).toMatchObject({
    allowed_callers: mode === 'hybrid' ? ['direct', 'code_execution'] : ['code_execution'],
    defer_loading: mode !== 'hybrid',
  });
});

it('distinguishes default and empty local selections and binds exactly one runner', () => {
  const defaults = createLocalProgrammaticProfile({ mode: 'hybrid' });
  expect(defaults.toolExecution.local?.codingToolNames).toEqual([
    ...createLocalCodingToolDefinitions().map((definition) => definition.name),
    Constants.BASH_TOOL, Constants.PROGRAMMATIC_TOOL_CALLING,
  ]);
  const empty = createLocalProgrammaticProfile({ mode: 'hybrid', local: { codingToolNames: [] } });
  expect(resolveLocalToolsForBinding(empty)?.map((entry) => 'name' in entry ? entry.name : undefined).sort()).toEqual(['run_tools_with_code', 'tool_search']);
  expect(() => createLocalProgrammaticProfile({ local: { codingToolNames: [Constants.PROGRAMMATIC_TOOL_CALLING] } })).toThrow('Unsupported profile local tool');
});

it.each([
  { allowed_callers: undefined, defer_loading: undefined },
  { allowed_callers: [] as t.AllowedCaller[], defer_loading: undefined },
  { allowed_callers: ['direct'] as t.AllowedCaller[], defer_loading: undefined },
  { allowed_callers: ['code_execution'] as t.AllowedCaller[], defer_loading: undefined },
  { allowed_callers: ['direct', 'code_execution'] as t.AllowedCaller[], defer_loading: true },
])('rejects incompatible local metadata instead of broadening a hybrid capability: %j', (policy) => {
  const definition: t.LCTool = { name: 'read_file', ...policy };
  expect(() => createLocalProgrammaticProfile({
    mode: 'hybrid', local: { codingToolNames: ['read_file'] },
    toolRegistry: new Map([['read_file', definition]]),
  })).toThrow(/does not permit|requires immediate availability/);
  expect(definition).toEqual({ name: 'read_file', ...policy });
});

it('copies compatible local caller permissions without replacing canonical schemas', () => {
  const definition: t.LCTool = { name: 'read_file', parameters, allowed_callers: ['direct', 'code_execution'] };
  const profile = createLocalProgrammaticProfile({
    mode: 'hybrid', local: { codingToolNames: ['read_file'] },
    toolRegistry: new Map([['read_file', definition]]),
  });
  expect(profile.toolRegistry.get('read_file')?.parameters).toHaveProperty('properties.path');
  expect(definition).toEqual({ name: 'read_file', parameters, allowed_callers: ['direct', 'code_execution'] });
});

it.each([undefined, 100])('supports unnumbered complete data and bounded windows (cap=%s)', async (cap) => {
  const cwd = await mkdtemp(join(tmpdir(), 'lc-plain-read-'));
  try {
    await writeFile(join(cwd, 'data.json'), '{\n"value":1\n}');
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: cap });
    expect(JSON.parse(String(await read.invoke({ path: 'data.json', line_numbers: false })))).toEqual({ value: 1 });
    expect(String(await read.invoke({ path: 'data.json' }))).toContain('     1\t{');
    expect(String(await read.invoke({ path: 'data.json', line_numbers: false, offset: 2, limit: 1 }))).toContain('"value":1');
    expect(String(await read.invoke({ path: 'data.json', line_numbers: false, offset: 2, limit: 1 }))).not.toContain('     2\t');
  } finally { await rm(cwd, { recursive: true, force: true }); }
});

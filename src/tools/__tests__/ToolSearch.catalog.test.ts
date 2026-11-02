import { tool } from '@langchain/core/tools';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import type { LCToolRegistry } from '@/types';
import { resolveLocalToolsForBinding } from '../local/resolveLocalExecutionTools';
import { createLocalProgrammaticProfile } from '../local/LocalProgrammaticProfile';
import { createToolSearch } from '../ToolSearch';

function registry(size: number, manyServers = false): LCToolRegistry {
  const tools: LCToolRegistry = new Map();
  for (let i = 0; i < size; i++) {
    const name = `archive_record_${i}_mcp_${manyServers ? `server_${i}` : 'archive'}`;
    tools.set(name, {
      name, description: `Read archived record ${i}`, defer_loading: true,
      allowed_callers: ['code_execution'],
      parameters: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
    });
  }
  tools.set('repair_build_mcp_ci', {
    name: 'repair_build_mcp_ci', description: 'Diagnose a failed continuous integration build',
    defer_loading: true, allowed_callers: ['code_execution'],
    parameters: {
      type: 'object', properties: { run: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] } },
      required: ['run'],
    },
  });
  return tools;
}

describe('bounded initial tool catalog', () => {
  it.each([false, true])('bounds the preview while keeping the complete search index (many servers: %s)', async (manyServers) => {
    const toolRegistry = registry(500, manyServers);
    const options = { mode: 'local' as const, includeSchemas: true, toolRegistry, maxCatalogChars: 256 };
    const bounded = createToolSearch(options);
    const control = createToolSearch({ ...options, maxCatalogChars: undefined });
    expect(bounded.description.length).toBeLessThan(600);
    expect(bounded.description).toContain('501 tools');
    expect(bounded.description).toContain('Names omitted');
    expect(bounded.description).not.toContain('archive_record_499');
    expect(control.description).toContain('archive_record_499');
    for (const query of ['archive_record_499', 'failed continuous integration build']) {
      const args = { query };
      expect(await bounded.invoke(args)).toEqual(await control.invoke(args));
    }
    const result = JSON.parse(String(await bounded.invoke({ query: 'repair_build' }))) as { tools: { parameters: object }[] };
    expect(result.tools[0].parameters).toEqual(toolRegistry.get('repair_build_mcp_ci')?.parameters);
  });

  it('keeps a small complete catalog and legacy defaults', () => {
    const toolRegistry = registry(2);
    const options = { mode: 'local' as const, toolRegistry, maxCatalogChars: 1024 };
    expect(createToolSearch(options).description).toEqual(createToolSearch({ mode: 'local', toolRegistry }).description);
  });

  it('keeps MCP guidance out of local-only catalogs while supporting unknown runtime catalogs', () => {
    const toolRegistry: LCToolRegistry = new Map([['read_file', { name: 'read_file', defer_loading: true }]]);
    for (const maxCatalogChars of [undefined, 128]) {
      expect(createToolSearch({ mode: 'local', toolRegistry, maxCatalogChars }).description).not.toContain('MCP tools use format');
    }
    expect(createToolSearch({ mode: 'local' }).description).toContain('MCP tools use format');
  });

  it('handles server names matching object prototype properties', () => {
    const toolRegistry: LCToolRegistry = new Map([['read_mcp___proto__', { name: 'read_mcp___proto__', defer_loading: true }]]);
    expect(createToolSearch({ mode: 'local', toolRegistry, maxCatalogChars: 128 }).description).toContain('__proto__: read');
  });

  it('keeps bounded previews deterministic across registry insertion orders', () => {
    const entries = [...registry(500, true)];
    const options = { mode: 'local' as const, maxCatalogChars: 256 };
    expect(createToolSearch({ ...options, toolRegistry: new Map(entries) }).description).toEqual(
      createToolSearch({ ...options, toolRegistry: new Map(entries.reverse()) }).description
    );
  });

  it('does not truncate server identifiers or expose non-deferred names', () => {
    const toolRegistry = registry(500);
    const longServer = 'long'.repeat(100);
    toolRegistry.set(`read_mcp_${longServer}`, { name: `read_mcp_${longServer}`, defer_loading: true });
    toolRegistry.set('internal_mcp_private', { name: 'internal_mcp_private', defer_loading: false });
    const options = { mode: 'local' as const, toolRegistry, maxCatalogChars: 128 };
    const search = createToolSearch(options);
    expect(search.description.length).toBeLessThan(500);
    expect(search.description).toContain('502 tools');
    expect(search.description).not.toContain('longlong');
    expect(search.description).not.toContain('private');
  });

  it.each([-1, 0, 127, 128.5, NaN, Infinity])('rejects invalid catalog budget %s', (maxCatalogChars) => {
    const options = { mode: 'local' as const, maxCatalogChars };
    expect(() => createToolSearch(options)).toThrow('maxCatalogChars');
  });

  it.each(['compact', 'hybrid'] as const)('keeps %s bound schemas small and unchanged after discovery', (mode) => {
    const toolRegistry = registry(500);
    const tools = [...toolRegistry.values()].map((definition) => tool(async () => 'fixture', {
      name: definition.name, description: definition.description, schema: definition.parameters ?? { type: 'object' },
    }));
    const options = { mode, tools, toolRegistry, maxCatalogChars: 256, local: { codingToolNames: [] } };
    const profile = createLocalProgrammaticProfile(options);
    const schemas = (discoveredToolNames: Set<string>) => (resolveLocalToolsForBinding({ ...profile, discoveredToolNames }) ?? [])
      .map((bound) => convertToOpenAITool(bound));
    const initial = schemas(new Set());
    const afterDiscovery = schemas(new Set(['repair_build_mcp_ci']));
    expect(afterDiscovery).toEqual(initial);
    const search = initial.find((entry) => entry.function.name === 'tool_search');
    expect(search?.function.description?.length).toBeLessThan(600);
    expect(search?.function.description).not.toContain('archive_record_499');
    expect(profile.toolRegistry.size).toBe(501);
  });
});

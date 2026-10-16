import { describe, it, expect } from '@jest/globals';
import type * as t from '@/types';
import {
  filterToolsByLocalAllowlist,
  isProgrammaticRunnerAutoBound,
  resolveLocalExecutionTools,
  resolveLocalImplementationNames,
  resolveLocalToolRegistry,
  resolveLocalToolsForBinding,
} from '../local/resolveLocalExecutionTools';
import { createLocalCodingTools } from '../local/LocalCodingTools';
import { INTENT_ARG } from '../intentArg';
import { Constants } from '@/common';

const propsOf = (schema: unknown): Record<string, unknown> =>
  (schema as { properties?: Record<string, unknown> }).properties ?? {};

describe('local coding tool selection', () => {
  it('binds the full suite with intent labels by default', () => {
    const tools = createLocalCodingTools();
    const read = tools.find((tool) => tool.name === Constants.READ_FILE);
    expect(tools.length).toBeGreaterThan(4);
    expect(propsOf(read?.schema)[INTENT_ARG]).toBeDefined();
  });

  it('keeps only the allowlisted tools', () => {
    const names = [
      Constants.READ_FILE,
      Constants.EDIT_FILE,
      Constants.BASH_TOOL,
    ];
    const tools = createLocalCodingTools({ codingToolNames: names });
    expect(tools.map((tool) => tool.name).sort()).toEqual([...names].sort());
  });

  it('drops the intent label from every schema when toolIntentLabels is false', () => {
    const tools = createLocalCodingTools({ toolIntentLabels: false });
    for (const tool of tools) {
      expect(propsOf(tool.schema)[INTENT_ARG]).toBeUndefined();
    }
    const edit = tools.find((tool) => tool.name === Constants.EDIT_FILE);
    expect(Object.keys(propsOf(edit?.schema)).length).toBeGreaterThan(0);
  });

  it('prunes registry definitions outside the local allowlist', () => {
    const toolExecution: t.ToolExecutionConfig = {
      engine: 'local',
      local: { codingToolNames: [Constants.READ_FILE] },
    };
    const registry = resolveLocalToolRegistry({ toolExecution });
    expect(registry?.has(Constants.READ_FILE)).toBe(true);
    expect(registry?.has(Constants.LIST_DIRECTORY)).toBe(false);
  });

  it('drops host-bound coding tools outside the local allowlist on every path', () => {
    const toolExecution: t.ToolExecutionConfig = {
      engine: 'local',
      local: { codingToolNames: [Constants.READ_FILE] },
    };
    const hostBash = { name: Constants.BASH_TOOL } as t.GenericTool;
    const bound = resolveLocalToolsForBinding({
      tools: [hostBash],
      toolExecution,
    }) as t.GenericTool[];
    expect(bound.map((tool) => tool.name)).toEqual([Constants.READ_FILE]);

    const { toolMap, localImplementationNames } = resolveLocalExecutionTools({
      toolMap: new Map([[Constants.BASH_TOOL, hostBash]]) as t.ToolMap,
      toolExecution,
    });
    expect([...toolMap.keys()]).toEqual([Constants.READ_FILE]);
    expect([...localImplementationNames]).toEqual([Constants.READ_FILE]);
    expect([
      ...resolveLocalImplementationNames([Constants.BASH_TOOL], toolExecution),
    ]).toEqual([Constants.READ_FILE]);
  });

  it('reports programmatic runners as auto-bound only when allowlisted', () => {
    const toolExecution: t.ToolExecutionConfig = {
      engine: 'local',
      local: { codingToolNames: [Constants.READ_FILE] },
    };
    expect(
      isProgrammaticRunnerAutoBound(
        Constants.PROGRAMMATIC_TOOL_CALLING,
        toolExecution
      )
    ).toBe(false);
    expect(
      isProgrammaticRunnerAutoBound(Constants.PROGRAMMATIC_TOOL_CALLING, {
        engine: 'local',
      })
    ).toBe(true);
  });

  it('filters host-bound tools for prompt projections with the same allowlist', () => {
    const toolExecution: t.ToolExecutionConfig = {
      engine: 'local',
      local: { codingToolNames: [Constants.READ_FILE] },
    };
    const hostTools = [
      { name: Constants.PROGRAMMATIC_TOOL_CALLING } as t.GenericTool,
      { name: 'web_search' } as t.GenericTool,
    ];
    const filtered = filterToolsByLocalAllowlist(
      hostTools,
      toolExecution
    ) as t.GenericTool[];
    expect(filtered.map((tool) => tool.name)).toEqual(['web_search']);
    expect(filterToolsByLocalAllowlist(hostTools, { engine: 'local' })).toBe(
      hostTools
    );
  });

  it('drops the intent label from replacement tools when coding tools are off', () => {
    const tools = resolveLocalToolsForBinding({
      tools: [{ name: Constants.BASH_TOOL } as t.GenericTool],
      toolExecution: {
        engine: 'local',
        local: { includeCodingTools: false, toolIntentLabels: false },
      },
    }) as t.GenericTool[];
    const bash = tools.find((tool) => tool.name === Constants.BASH_TOOL);
    expect(propsOf((bash as { schema?: unknown }).schema)[INTENT_ARG]).toBeUndefined();
  });

  it('filters event-driven tool definitions with the local allowlist', () => {
    const definitions = [
      { name: Constants.READ_FILE },
      { name: Constants.WRITE_FILE },
      { name: 'web_search' },
    ] as t.LCTool[];
    const filtered = filterToolsByLocalAllowlist(
      definitions as unknown as t.GraphTools,
      { engine: 'local', local: { codingToolNames: [Constants.READ_FILE] } }
    ) as unknown as t.LCTool[];
    expect(filtered.map((def) => def.name)).toEqual([
      Constants.READ_FILE,
      'web_search',
    ]);
  });
});

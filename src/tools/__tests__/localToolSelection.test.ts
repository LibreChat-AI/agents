import { describe, it, expect } from '@jest/globals';
import type * as t from '@/types';
import { resolveLocalToolRegistry } from '../local/resolveLocalExecutionTools';
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
});

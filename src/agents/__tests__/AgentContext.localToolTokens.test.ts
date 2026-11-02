import { SystemMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import type * as t from '@/types';
import { resolveLocalToolsForBinding } from '@/tools/local/resolveLocalExecutionTools';
import { createLocalProgrammaticProfile } from '@/tools/local/LocalProgrammaticProfile';
import { Providers } from '@/common';
import { AgentContext } from '../AgentContext';

const counter: t.TokenCounter = (message) => typeof message.content === 'string'
  ? Math.ceil(message.content.length / 4) + 4 : 4;

function context(toolExecution?: t.ToolExecutionConfig, tools?: t.GraphTools,
  toolRegistry?: t.LCToolRegistry, cachedTokens?: number): AgentContext {
  return AgentContext.fromConfig({ agentId: 'local-token-test', provider: Providers.OPENAI,
    instructions: 'Read and edit the repository.', tools, toolRegistry, toolSchemaTokens: cachedTokens }, counter, {}, toolExecution);
}

describe('automatically bound local tool accounting', () => {
  it.each([
    { local: {} },
    { local: { codingToolNames: ['read_file', 'bash_tool'] } },
    { local: { codingToolNames: [] } },
    { local: { includeCodingTools: false } },
  ])('counts the same schemas as an explicitly installed equivalent suite: %j', async (config) => {
    const execution: t.ToolExecutionConfig = { engine: 'local', ...config };
    const auto = context(execution);
    const bound = resolveLocalToolsForBinding({ toolExecution: execution });
    const explicit = context(undefined, bound);
    await Promise.all([auto.tokenCalculationPromise, explicit.tokenCalculationPromise]);
    expect(auto.toolSchemaTokens).toBe(explicit.toolSchemaTokens);
    expect(auto.toolTokenCounts).toEqual(explicit.toolTokenCounts);
    expect(auto.getTokenBudgetBreakdown().toolCount).toBe(bound?.length ?? 0);
    if (config.local.codingToolNames?.length !== 0 && config.local.includeCodingTools !== false) {
      expect(auto.toolSchemaTokens).toBeGreaterThan(0);
    }
    expect(auto.instructionTokens).toBeGreaterThanOrEqual(counter(new SystemMessage('Read and edit the repository.')));
  });

  it.each(['compact', 'hybrid'] as const)('keeps %s programmatic profile accounting aligned with bound tools', async (mode) => {
    const profile = createLocalProgrammaticProfile({ mode });
    const auto = context(profile.toolExecution, profile.tools, profile.toolRegistry);
    const bound = resolveLocalToolsForBinding({ tools: auto.getToolsForBinding(),
      toolExecution: profile.toolExecution, toolRegistry: auto.toolRegistry });
    const explicit = context(undefined, bound);
    await Promise.all([auto.tokenCalculationPromise, explicit.tokenCalculationPromise]);
    expect(auto.toolSchemaTokens).toBe(explicit.toolSchemaTokens);
    expect(Object.keys(auto.toolTokenCounts ?? {}).sort()).toEqual(Object.keys(explicit.toolTokenCounts ?? {}).sort());
    auto.reset();
    await auto.tokenCalculationPromise;
    expect(auto.toolSchemaTokens).toBe(explicit.toolSchemaTokens);
  });

  it('adds new local schemas to an opaque cached host total', async () => {
    const execution: t.ToolExecutionConfig = { engine: 'local', local: { codingToolNames: ['read_file'] } };
    const measured = context(execution);
    const cached = context(execution, undefined, undefined, 10000);
    await Promise.all([measured.tokenCalculationPromise, cached.tokenCalculationPromise]);
    expect(cached.toolSchemaTokens).toBe(10000 + measured.toolSchemaTokens);
  });

  it('keeps cached totals unchanged when local resolution adds no schemas', async () => {
    const execution: t.ToolExecutionConfig = { engine: 'local', local: { codingToolNames: ['read_file'] } };
    const bound = resolveLocalToolsForBinding({ toolExecution: execution });
    const alreadyInstalled = context(execution, bound, undefined, 10000);
    const disabled = context({ engine: 'local', local: { includeCodingTools: false } }, undefined, undefined, 10000);
    await Promise.all([alreadyInstalled.tokenCalculationPromise, disabled.tokenCalculationPromise]);
    expect(alreadyInstalled.toolSchemaTokens).toBe(10000);
    expect(disabled.toolSchemaTokens).toBe(10000);
  });

  it('adds only positive schema growth for a measurable replacement', async () => {
    const execution: t.ToolExecutionConfig = { engine: 'local', local: { codingToolNames: ['read_file'] } };
    const original = tool(async () => 'unused', { name: 'read_file', description: 'read',
      schema: { type: 'object', properties: {} } });
    const prior = context(undefined, [original]);
    const measured = context(execution);
    const cached = context(execution, [original], undefined, 10000);
    await Promise.all([prior.tokenCalculationPromise, measured.tokenCalculationPromise, cached.tokenCalculationPromise]);
    const growth = measured.toolSchemaTokens - prior.toolSchemaTokens;
    expect(growth).toBeGreaterThan(0);
    expect(cached.toolSchemaTokens).toBeGreaterThanOrEqual(10000 + growth);
    expect(cached.toolSchemaTokens).toBeLessThanOrEqual(10001 + growth);
  });
});

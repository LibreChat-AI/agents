import type * as t from '@/types';
import { Constants, Providers } from '@/common';
import { AgentContext } from '../AgentContext';

type Style = t.ToolExecutionConfig['programmaticToolInstructions'];

function inputs(style?: Style): t.AgentInputs {
  const toolDefinitions: t.LCTool[] = [
    { name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING },
    { name: Constants.TOOL_SEARCH },
    {
      name: 'semantic_lookup',
      description: 'Look up an item',
      allowed_callers: ['code_execution'],
      parameters: { type: 'object', properties: {} },
    },
  ];
  return {
    agentId: 'test-agent',
    provider: Providers.OPENAI,
    instructions: 'Base instructions',
    programmaticToolInstructions: style,
    toolDefinitions,
    toolRegistry: new Map(toolDefinitions.map((tool) => [tool.name, tool])),
  };
}

async function prompt(context: AgentContext): Promise<string> {
  const messages = await context.systemRunnable?.invoke([]);
  const content = messages?.[0]?.content;
  if (typeof content !== 'string')
    throw new Error('Expected a text system prompt');
  return content;
}

describe('agent programmatic instruction policy', () => {
  it.each([
    ['discovery', 'inline', true],
    ['inline', 'discovery', false],
    [undefined, 'discovery', true],
    [undefined, 'inline', false],
    [undefined, undefined, false],
  ] as const)(
    'agent %s, run %s selects discovery=%s without changing binding',
    async (agentStyle, runStyle, discovery) => {
      const execution: t.ToolExecutionConfig = {
        engine: 'sandbox',
        programmaticToolInstructions: runStyle,
      };
      const before = { ...execution };
      const context = AgentContext.fromConfig(
        inputs(agentStyle),
        undefined,
        undefined,
        execution
      );
      const text = await prompt(context);
      expect(text.includes('Discover callable tools with tool_search')).toBe(
        discovery
      );
      expect(text.includes('semantic_lookup')).toBe(!discovery);
      expect(context.getToolsForBinding()).toEqual([
        expect.objectContaining({ name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING }),
        expect.objectContaining({ name: Constants.TOOL_SEARCH }),
      ]);
      expect(execution).toEqual(before);
    }
  );

  it('keeps policies independent between agents sharing one run configuration', async () => {
    const execution: t.ToolExecutionConfig = { engine: 'sandbox' };
    const parent = AgentContext.fromConfig(
      inputs('discovery'),
      undefined,
      undefined,
      execution
    );
    const child = AgentContext.fromConfig(
      inputs('inline'),
      undefined,
      undefined,
      execution
    );
    expect(await prompt(parent)).not.toContain('semantic_lookup');
    expect(await prompt(child)).toContain('semantic_lookup');
    expect(execution).toEqual({ engine: 'sandbox' });
  });

  it('honors explicit hybrid guidance in the direct constructor', async () => {
    const context = new AgentContext({
      ...inputs('hybrid'),
      toolExecution: {
        engine: 'sandbox',
        programmaticToolInstructions: 'inline',
      },
    });
    expect(await prompt(context)).toContain(
      'The profile\'s selected local coding tools'
    );
    expect(await prompt(context)).not.toContain('semantic_lookup');
  });
});

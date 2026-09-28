import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT } from '@/langfuseToolOutputTracing';
import { Providers } from '@/common';
import { Run } from '@/run';

const invoke = jest.fn();

jest.mock('@/llm/init', () => ({
  initializeModel: jest.fn(() => ({ invoke })),
}));

const REDACTING_POLICY: t.LangfuseConfig = {
  toolOutputTracing: {
    redactedToolNames: ['run_select_query'],
    redactedToolNameMatchMode: 'partial',
  },
};

async function createRun(): Promise<Run<never>> {
  const run = await Run.create({
    runId: 'redaction-scope-run',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent-1',
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4.1-mini' },
          tools: [],
        },
      ],
    },
    langfuse: REDACTING_POLICY,
  });
  if (run.Graph != null) {
    run.Graph.messages = [new HumanMessage('How many EU orders came in?')];
  }
  return run;
}

function modelPrompt(): string {
  const messages = invoke.mock.calls[0][0] as AIMessage[];
  return String(messages[1].content);
}

/** Tool-output redaction scopes to tool-call observations; label calls are
 *  written from the full evidence under any policy. */
describe('label evidence under a tool-output redaction policy', () => {
  beforeEach(() => {
    invoke.mockReset();
    invoke.mockResolvedValue(new AIMessage('Counted EU orders by region'));
  });

  it('writes an activity label from the redacted tool output and reasoning', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityLabel({
        provider: Providers.OPENAI,
        entries: [
          {
            toolName: 'run_select_query',
            toolInput: { query: 'SELECT count() FROM orders' },
            toolOutput: 'EU_ROWS_42113',
            status: 'success',
          },
        ],
        thinkingExcerpts: ['Checking the EU order count'],
        previousLabels: ['Listed the orders tables'],
      })
    ).resolves.toEqual({ label: 'Counted EU orders by region' });

    expect(modelPrompt()).toContain('EU_ROWS_42113');
    expect(modelPrompt()).toContain('Checking the EU order count');
    expect(modelPrompt()).toContain('Listed the orders tables');
    expect(modelPrompt()).not.toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('labels a reasoning-only block', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityLabel({
        provider: Providers.OPENAI,
        entries: [],
        thinkingExcerpts: ['Comparing EU and US order volume'],
      })
    ).resolves.toEqual({ label: 'Counted EU orders by region' });
    expect(modelPrompt()).toContain('Comparing EU and US order volume');
  });

  it('titles a phase of labeled activities', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [
          { agentId: 'agent-1', label: 'Listed the orders tables' },
          { agentId: 'agent-1', label: 'Measured daily order volume' },
        ],
        assistantContext: ['Comparing regions next'],
      })
    ).resolves.toEqual({ label: 'Counted EU orders by region' });

    expect(modelPrompt()).toContain('Listed the orders tables');
    expect(modelPrompt()).toContain('Measured daily order volume');
    expect(modelPrompt()).toContain('Comparing regions next');
  });

  it('labels visible reasoning', async () => {
    const run = await createRun();

    const result = await run.generateReasoningLabel({
      provider: Providers.OPENAI,
      visibleReasoning: 'The query returned EU_ROWS_42113, so EU leads',
      reasoningStepId: 'reasoning-step-1',
      revision: 0,
    });

    expect(result.label).toBe('Counted EU orders by region');
    expect(modelPrompt()).toContain('EU_ROWS_42113');
  });
});

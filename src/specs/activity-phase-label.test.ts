import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT } from '@/langfuseToolOutputTracing';
import { Providers } from '@/common';
import { Run } from '@/run';

const invoke = jest.fn();

jest.mock('@/llm/init', () => ({
  initializeModel: jest.fn(() => ({ invoke })),
}));

async function createRun(): Promise<Run<never>> {
  return Run.create({
    runId: 'phase-run',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent-1',
          name: 'Phase Agent',
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4.1-mini' },
          tools: [],
        },
      ],
    },
  });
}

describe('generateActivityPhaseLabel', () => {
  beforeEach(() => {
    invoke.mockReset();
    invoke.mockResolvedValue(
      new AIMessage('"Fixed session refresh handling and verified auth tests."')
    );
  });

  it('does not spend a model call on one logical activity', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [{ label: 'Inspected session refresh middleware' }],
      })
    ).resolves.toEqual({});
    expect(invoke).not.toHaveBeenCalled();
  });

  it('summarizes two activities and normalizes the persisted row', async () => {
    const run = await createRun();
    if (run.Graph != null) {
      run.Graph.messages = [
        new HumanMessage({
          content: [
            {
              type: 'input_text',
              text: 'Why is session refresh failing?',
            } as never,
          ],
        }),
        new HumanMessage({
          content: 'Internal routing instructions',
          additional_kwargs: {
            role: 'user',
            isMeta: true,
            source: 'routing',
          },
        }),
      ];
    }
    const handleChainStart = jest.fn();
    const handleChainEnd = jest.fn();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [
          { label: 'Inspected session refresh middleware' },
          { label: 'Fixed refresh token validation' },
        ],
        assistantContext: ['I am checking the auth path.'],
        closingTextPhase: 'final_answer',
        chainOptions: {
          callbacks: [{ handleChainStart, handleChainEnd }],
          configurable: {
            requestBody: { parentMessageId: 'parent-message-1' },
          },
        },
      })
    ).resolves.toEqual({
      label: 'Fixed session refresh handling and verified auth tests',
    });
    expect(invoke).toHaveBeenCalledTimes(1);
    const messages = invoke.mock.calls[0][0] as AIMessage[];
    expect(String(messages[1].content)).toContain(
      'Inspected session refresh middleware'
    );
    expect(String(messages[1].content)).toContain(
      'Fixed refresh token validation'
    );
    expect(String(messages[1].content)).not.toContain(
      'Why is session refresh failing?'
    );
    const modelConfig = invoke.mock.calls[0][1] as {
      callbacks?: { getParentRunId?: () => string | undefined };
      tags?: string[];
      metadata?: Record<string, unknown>;
    };
    expect(modelConfig.callbacks?.getParentRunId?.()).toEqual(
      expect.any(String)
    );
    expect(modelConfig.tags).toEqual(
      expect.arrayContaining(['activity-phase', 'agent'])
    );
    expect(modelConfig.metadata).toEqual(
      expect.objectContaining({
        agentId: 'agent-1',
        agentName: 'Phase Agent',
        parentMessageId: 'parent-message-1',
      })
    );
    expect(handleChainStart.mock.calls[0]?.[1]).toEqual(
      expect.objectContaining({
        messages: expect.arrayContaining([
          expect.objectContaining({
            content: 'Why is session refresh failing?',
          }),
        ]),
      })
    );
    expect(JSON.stringify(handleChainStart.mock.calls[0]?.[1])).not.toContain(
      'Internal routing instructions'
    );
    expect(handleChainEnd.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        messages: expect.arrayContaining([
          expect.objectContaining({
            content: 'Fixed session refresh handling and verified auth tests',
          }),
        ]),
      })
    );
  });

  it('does not call the model when retained activities have no evidence', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [
          ...Array.from({ length: 12 }, () => ({ status: 'success' as const })),
          { label: 'Evidence beyond the prompt cap' },
        ],
      })
    ).resolves.toEqual({});
    expect(invoke).not.toHaveBeenCalled();
  });

  it('preserves the phase parent when retrying without a failing callback', async () => {
    invoke.mockRejectedValueOnce(new Error('event stream callback failed'));
    const run = await createRun();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [
          { label: 'Inspected session refresh middleware' },
          { label: 'Fixed refresh token validation' },
        ],
        chainOptions: {
          callbacks: [{ handleChainStart: jest.fn() }],
        },
      })
    ).resolves.toEqual({
      label: 'Fixed session refresh handling and verified auth tests',
    });

    expect(invoke).toHaveBeenCalledTimes(2);
    const firstCallbacks = invoke.mock.calls[0][1].callbacks as {
      getParentRunId: () => string | undefined;
      handlers: unknown[];
    };
    const retryCallbacks = invoke.mock.calls[1][1].callbacks as {
      getParentRunId: () => string | undefined;
      handlers: unknown[];
    };
    expect(firstCallbacks.getParentRunId()).toEqual(expect.any(String));
    expect(retryCallbacks.getParentRunId()).toBe(
      firstCallbacks.getParentRunId()
    );
    expect(retryCallbacks.handlers).toHaveLength(0);
  });

  it('sends every agent\'s evidence to the model when an activity is unattributed', async () => {
    const run = await Run.create({
      runId: 'mixed-attribution-phase-run',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent-1',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
          },
          {
            agentId: 'agent-2',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
            langfuse: {
              toolOutputTracing: { redactedToolNames: ['secret_tool'] },
            },
          },
        ],
        edges: [],
      },
    });

    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        { agentId: 'agent-1', label: 'Inspected public session behavior' },
        {
          entries: [
            {
              toolName: 'secret_tool',
              toolInput: { key: 'public-key' },
              toolOutput: 'STRICT_AGENT_SECRET',
              status: 'success',
            },
          ],
        },
      ],
    });

    /** Tool-output redaction governs tool-call observations, not label evidence. */
    const messages = invoke.mock.calls[0][0] as AIMessage[];
    expect(String(messages[1].content)).toContain('STRICT_AGENT_SECRET');
    expect(String(messages[1].content)).not.toContain(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
  });

  it('keeps assistant context for the model when omitted activities have no agent list', async () => {
    const run = await Run.create({
      runId: 'omitted-attribution-phase-run',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent-1',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
          },
          {
            agentId: 'agent-2',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
            langfuse: {
              toolOutputTracing: { redactedToolNames: ['secret_tool'] },
            },
          },
        ],
        edges: [],
      },
    });

    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        {
          agentId: 'agent-1',
          entries: [
            {
              toolName: 'public_lookup',
              toolInput: { id: 'one' },
              toolOutput: 'public-one',
              status: 'success',
            },
          ],
        },
        {
          agentId: 'agent-1',
          entries: [
            {
              toolName: 'public_lookup',
              toolInput: { id: 'two' },
              toolOutput: 'public-two',
              status: 'success',
            },
          ],
        },
      ],
      totalActivityCount: 3,
      assistantContext: ['OMITTED_AGENT_SECRET'],
    });

    const messages = invoke.mock.calls[0][0] as AIMessage[];
    expect(String(messages[1].content)).toContain('OMITTED_AGENT_SECRET');
    expect(String(messages[1].content)).toContain('public-one');
  });
});

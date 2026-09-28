import { AIMessage } from '@langchain/core/messages';
import { Providers } from '@/common';
import { Run } from '@/run';

const invoke = jest.fn();

jest.mock('@/llm/init', () => ({
  initializeModel: jest.fn(() => ({ invoke })),
}));

const usage = {
  input_tokens: 21,
  output_tokens: 6,
  total_tokens: 27,
};

async function createRun(): Promise<Run<never>> {
  return Run.create({
    runId: 'reasoning-run',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent-1',
          name: 'Reasoning Agent',
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4.1-mini' },
          tools: [],
        },
      ],
    },
  });
}

describe('generateReasoningLabel', () => {
  beforeEach(() => {
    invoke.mockReset();
    invoke.mockResolvedValue(
      new AIMessage({
        content: '"Tracing refresh failures through middleware."',
        usage_metadata: usage,
      })
    );
  });

  it('generates a bounded replacement title and returns provider usage', async () => {
    const run = await createRun();

    await expect(
      run.generateReasoningLabel({
        provider: Providers.OPENAI,
        agentId: 'agent-1',
        visibleReasoning:
          'I am following the refresh request through each middleware layer.',
        previousLabel: 'Inspecting the authentication path',
        reasoningStepId: 'reasoning-step-1',
        revision: 2,
        status: 'streaming',
        sourceRunId: 'source-run-1',
        sourceTraceId: 'source-trace-1',
        responseId: 'response-1',
        chainOptions: {
          configurable: {
            requestBody: { parentMessageId: 'parent-message-1' },
          },
        },
      })
    ).resolves.toEqual({
      label: 'Tracing refresh failures through middleware',
      usage,
    });

    expect(invoke).toHaveBeenCalledTimes(1);
    const messages = invoke.mock.calls[0][0] as AIMessage[];
    expect(String(messages[1].content)).toContain(
      'following the refresh request'
    );
    expect(String(messages[1].content)).toContain(
      'Inspecting the authentication path'
    );
    const config = invoke.mock.calls[0][1] as {
      runId?: string;
      tags?: string[];
      metadata?: Record<string, unknown>;
    };
    expect(config.runId).toBe('reasoning-run-reasoning-1');
    expect(config.tags).toEqual(
      expect.arrayContaining(['reasoning-label', 'reasoning-step'])
    );
    expect(config.metadata).toEqual(
      expect.objectContaining({
        sourceRunId: 'source-run-1',
        sourceTraceId: 'source-trace-1',
        responseId: 'response-1',
        reasoningStepId: 'reasoning-step-1',
        revision: 2,
        status: 'streaming',
        parentMessageId: 'parent-message-1',
        agentId: 'agent-1',
        agentName: 'Reasoning Agent',
      })
    );
  });

  it('correlates revisions by step while keeping invocation ids distinct', async () => {
    const run = await createRun();
    for (const revision of [0, 1]) {
      await run.generateReasoningLabel({
        provider: Providers.OPENAI,
        agentId: 'agent-1',
        visibleReasoning: `Visible snapshot revision ${revision}`,
        reasoningStepId: 'stable-step',
        revision,
      });
    }

    expect(invoke).toHaveBeenCalledTimes(2);
    const configs = invoke.mock.calls.map(
      (call) =>
        call[1] as {
          runId: string;
          metadata: Record<string, unknown>;
        }
    );
    expect(configs.map((config) => config.runId)).toEqual([
      'reasoning-run-reasoning-1',
      'reasoning-run-reasoning-2',
    ]);
    expect(
      configs.map((config) => ({
        reasoningStepId: config.metadata.reasoningStepId,
        revision: config.metadata.revision,
      }))
    ).toEqual([
      { reasoningStepId: 'stable-step', revision: 0 },
      { reasoningStepId: 'stable-step', revision: 1 },
    ]);
  });

  it('fails closed for an explicit unknown executing agent', async () => {
    const run = await createRun();

    await expect(
      run.generateReasoningLabel({
        provider: Providers.OPENAI,
        agentId: 'missing-agent',
        visibleReasoning: 'Checking a private agent result',
        reasoningStepId: 'reasoning-step-1',
        revision: 0,
      })
    ).resolves.toEqual({});
    expect(invoke).not.toHaveBeenCalled();
  });

  it('labels the reasoning snapshot under the executing agent policy', async () => {
    const run = await Run.create({
      runId: 'redacted-reasoning-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'strict-agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
            langfuse: {
              toolOutputTracing: { redactedToolNames: ['secret_lookup'] },
            },
          },
        ],
      },
    });

    await expect(
      run.generateReasoningLabel({
        provider: Providers.OPENAI,
        agentId: 'strict-agent',
        visibleReasoning: 'The secret lookup returned a private credential',
        reasoningStepId: 'reasoning-step-1',
        revision: 0,
      })
    ).resolves.toHaveProperty('label');
    /** Tool-output redaction governs tool-call observations, not label evidence. */
    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it('rejects invalid step identity and revision without calling a model', async () => {
    const run = await createRun();

    for (const options of [
      { reasoningStepId: '', revision: 0 },
      { reasoningStepId: 'step', revision: -1 },
      { reasoningStepId: 'step', revision: 1.5 },
    ]) {
      await expect(
        run.generateReasoningLabel({
          provider: Providers.OPENAI,
          visibleReasoning: 'Visible reasoning',
          ...options,
        })
      ).resolves.toEqual({});
    }
    expect(invoke).not.toHaveBeenCalled();
  });

  it('does not retry callback failures after a provider invocation', async () => {
    invoke.mockRejectedValueOnce(new Error('event stream callback failed'));
    const run = await createRun();

    await expect(
      run.generateReasoningLabel({
        provider: Providers.OPENAI,
        visibleReasoning: 'Following the middleware chain',
        reasoningStepId: 'reasoning-step-1',
        revision: 0,
        chainOptions: {
          callbacks: [{ handleChainStart: jest.fn() }],
        },
      })
    ).rejects.toThrow('event stream callback failed');

    expect(invoke).toHaveBeenCalledTimes(1);
    expect(invoke.mock.calls[0][1].callbacks).toHaveLength(1);
  });

  it('does not retry provider failures', async () => {
    invoke.mockRejectedValueOnce(new Error('provider request failed'));
    const run = await createRun();

    await expect(
      run.generateReasoningLabel({
        provider: Providers.OPENAI,
        visibleReasoning: 'Following the middleware chain',
        reasoningStepId: 'reasoning-step-1',
        revision: 0,
      })
    ).rejects.toThrow('provider request failed');
    expect(invoke).toHaveBeenCalledTimes(1);
  });
});

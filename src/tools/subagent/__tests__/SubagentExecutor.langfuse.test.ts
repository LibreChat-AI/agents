import { createHash } from 'crypto';
import { AIMessage } from '@langchain/core/messages';
import { describe, expect, it, jest, afterEach } from '@jest/globals';
import { ROOT_CONTEXT, context as otelContext } from '@opentelemetry/api';
import type { StandardGraph } from '@/graphs/Graph';
import type { StandardGraphInput } from '@/types';
import { SubagentExecutor } from '../SubagentExecutor';
import { Providers } from '@/common';

function traceIdFromSeed(seed: string): string {
  return createHash('sha256').update(seed, 'utf8').digest('hex').slice(0, 32);
}

describe('SubagentExecutor linked Langfuse child trace', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('invokes the child graph as a detached AgentRun with parent link metadata', async () => {
    let capturedInvokeOptions: Record<string, unknown> | undefined;
    const withSpy = jest.spyOn(otelContext, 'with');

    const createChildGraph = (_input: StandardGraphInput) =>
      ({
        createWorkflow: () => ({
          invoke: jest.fn().mockImplementation(async (_state, options) => {
            capturedInvokeOptions = options as Record<string, unknown>;
            return { messages: [new AIMessage('child done')] };
          }),
        }),
        clearHeavyState: jest.fn(),
      }) as unknown as StandardGraph;

    const parentRunId = 'parent-message-id';
    const executor = new SubagentExecutor({
      configs: new Map([
        [
          'researcher',
          {
            type: 'researcher',
            name: 'Research Specialist',
            description: 'Researches things',
            agentInputs: {
              agentId: 'agent_researcher',
              name: 'Research Specialist',
              provider: Providers.OPENAI,
            },
          },
        ],
      ]),
      parentRunId,
      parentAgentId: 'agent_orchestrator',
      createChildGraph,
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
        secretKey: 'sk-test',
        baseUrl: 'https://langfuse.test',
        deterministicTraceId: true,
      },
    });

    const result = await executor.execute({
      description: 'Look up the invoice',
      subagentType: 'researcher',
      parentToolCallId: 'call_parent_subagent',
      parentConfigurable: {
        thread_id: 'conversation-session',
        user_id: 'user-123',
      },
    });

    expect(result.content).toBe('child done');
    expect(capturedInvokeOptions).toBeDefined();
    expect(capturedInvokeOptions?.runName).toBe('AgentRun');
    expect(withSpy).toHaveBeenCalledWith(ROOT_CONTEXT, expect.any(Function));

    const metadata = capturedInvokeOptions?.metadata as Record<string, string>;
    expect(metadata.parentMessageId).toBe(parentRunId);
    expect(metadata.parentToolCallId).toBe('call_parent_subagent');
    expect(metadata.subagentType).toBe('researcher');
    expect(metadata.parentTraceId).toBe(traceIdFromSeed(parentRunId));

    const configurable = capturedInvokeOptions?.configurable as Record<
      string,
      unknown
    >;
    expect(configurable.thread_id).toBe('conversation-session');
    expect(configurable.user_id).toBe('user-123');
  });

  it('appends host content parts onto the child HumanMessage', async () => {
    let capturedInput: { messages?: Array<{ content?: unknown }> } | undefined;
    const createChildGraph = (_input: StandardGraphInput) =>
      ({
        createWorkflow: () => ({
          invoke: jest.fn().mockImplementation(async (state) => {
            capturedInput = state as {
              messages?: Array<{ content?: unknown }>;
            };
            return { messages: [new AIMessage('saw media')] };
          }),
        }),
        clearHeavyState: jest.fn(),
      }) as unknown as StandardGraph;

    const executor = new SubagentExecutor({
      configs: new Map([
        [
          'researcher',
          {
            type: 'researcher',
            name: 'Research Specialist',
            description: 'Researches things',
            agentInputs: {
              agentId: 'agent_researcher',
              provider: Providers.OPENAI,
            },
          },
        ],
      ]),
      parentRunId: 'parent-run',
      createChildGraph,
    });

    await executor.execute({
      description: 'Read this PDF',
      subagentType: 'researcher',
      parentToolCallId: 'call_media',
      getAdditionalContentParts: () => [{ type: 'file', file_id: 'file_123' }],
    });

    const content = capturedInput?.messages?.[0]?.content;
    expect(content).toEqual([
      { type: 'text', text: 'Read this PDF' },
      { type: 'file', file_id: 'file_123' },
    ]);
  });
});

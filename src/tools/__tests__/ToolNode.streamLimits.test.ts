/**
 * `StreamLimitExceededError` must pass through ToolNode's tool-error
 * conversion like a `GraphInterrupt`: a child run (subagent) that trips a
 * stream circuit breaker is a safety abort, and converting it into an error
 * ToolMessage would let the parent keep generating after the limit fired.
 */
import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { ToolNode } from '@/tools/ToolNode';

const stateWith = (name: string): { messages: AIMessage[] } => ({
  messages: [
    new AIMessage({
      content: '',
      tool_calls: [{ id: 'call_1', name, args: { command: 'x' } }],
    }),
  ],
});

describe('ToolNode stream-limit passthrough', () => {
  it('rethrows StreamLimitExceededError instead of converting it to a ToolMessage', async () => {
    const limitBoom = tool(
      async () => {
        throw new StreamLimitExceededError({
          kind: 'tool_call_args',
          limit: 10,
          observed: 11,
          toolName: 'db_query',
        });
      },
      {
        name: 'limitBoom',
        description: 'tool that trips a stream limit',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;

    const node = new ToolNode({ tools: [limitBoom] });
    await expect(
      node.invoke(stateWith('limitBoom'), {
        configurable: { run_id: 'limit-run' },
      })
    ).rejects.toBeInstanceOf(StreamLimitExceededError);
  });

  it.each(['blocked', 'timeout'] as const)('rethrows child protection errors without error-handler conversion (%s)', async (code) => {
    const trip = new ProviderTextProtectionError(code);
    let errorHandled = false;
    const child = tool(async () => { throw trip; }, { name: 'child', description: 'Protected child.', schema: z.object({ command: z.string() }) });
    const node = new ToolNode({ tools: [child], errorHandler: async () => { errorHandled = true; return true; } });
    await expect(node.invoke(stateWith('child'), { configurable: { run_id: 'protected-child' } })).rejects.toBe(trip);
    expect(errorHandled).toBe(false);
  });

  it('still converts ordinary tool errors to error ToolMessages', async () => {
    const plainBoom = tool(
      async () => {
        throw new Error('ordinary failure');
      },
      {
        name: 'plainBoom',
        description: 'tool that throws a normal error',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;

    const node = new ToolNode({ tools: [plainBoom] });
    const result = (await node.invoke(stateWith('plainBoom'), {
      configurable: { run_id: 'plain-run' },
    })) as { messages: Array<{ content: unknown; status?: string }> };
    expect(result.messages).toHaveLength(1);
    expect(String(result.messages[0].content)).toContain('ordinary failure');
  });
});

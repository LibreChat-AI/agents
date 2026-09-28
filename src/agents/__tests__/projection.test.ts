import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { projectAgentContextUsage } from '../projection';
import { Providers } from '@/common';

const countByChars = (msg: { content: unknown }): number => {
  const content =
    typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
  return content.length;
};

const agent = (maxContextTokens: number): t.AgentInputs => ({
  agentId: 'test-agent',
  provider: Providers.OPENAI,
  instructions: 'system prompt',
  maxContextTokens,
});

const branch = (perMessageChars: number, count: number): AIMessage[] => {
  const messages: AIMessage[] = [];
  for (let i = 0; i < count; i++) {
    const content = 'x'.repeat(perMessageChars);
    messages.push(
      i % 2 === 0
        ? (new HumanMessage(content) as unknown as AIMessage)
        : new AIMessage(content)
    );
  }
  return messages;
};

describe('projectAgentContextUsage', () => {
  it('returns a budget snapshot for a branch that fits', async () => {
    const usage = await projectAgentContextUsage({
      agent: agent(100_000),
      messages: branch(1_000, 4),
      tokenCounter: countByChars,
    });

    expect(usage).not.toBeNull();
    expect(usage!.breakdown.maxContextTokens).toBe(100_000);
    expect(usage!.breakdown.messageCount).toBe(4);
    expect(usage!.remainingContextTokens).toBeGreaterThan(0);
    expect(usage!.agentId).toBe('test-agent');
  });

  it('prunes when the branch exceeds the window', async () => {
    const usage = await projectAgentContextUsage({
      agent: agent(3_000),
      messages: branch(1_000, 6),
      tokenCounter: countByChars,
    });

    expect(usage).not.toBeNull();
    expect(usage!.breakdown.messageCount).toBeGreaterThan(0);
    expect(usage!.breakdown.messageCount).toBeLessThan(6);
  });

  it('returns null without a context window', async () => {
    const noWindow: t.AgentInputs = {
      agentId: 'test-agent',
      provider: Providers.OPENAI,
      instructions: 'sys',
    };
    const usage = await projectAgentContextUsage({
      agent: noWindow,
      messages: branch(100, 2),
      tokenCounter: countByChars,
    });

    expect(usage).toBeNull();
  });

  it('projects instructions from the effective execution backend', async () => {
    const config: t.AgentInputs = {
      ...agent(100_000),
      toolRegistry: new Map([
        [
          'host_code_tool',
          { name: 'host_code_tool', allowed_callers: ['code_execution'] },
        ],
      ]),
    };
    const withoutExecution = await projectAgentContextUsage({
      agent: config,
      messages: [],
      tokenCounter: countByChars,
    });
    const withExecution = await projectAgentContextUsage({
      agent: config,
      messages: [],
      tokenCounter: countByChars,
      toolExecution: { engine: 'local' },
    });

    expect(withExecution!.breakdown.instructionTokens).toBeGreaterThan(
      withoutExecution!.breakdown.instructionTokens
    );
  });

  it('applies a persisted fading tier to the projected branch', async () => {
    const messages = [
      new HumanMessage('fetch the report'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call-1', name: 'fetch', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'x'.repeat(60_000),
        tool_call_id: 'call-1',
      }),
      new AIMessage('report received'),
    ];
    const withoutTier = await projectAgentContextUsage({
      agent: agent(100_000),
      messages,
      tokenCounter: countByChars,
    });
    const withTier = await projectAgentContextUsage({
      agent: agent(100_000),
      messages,
      tokenCounter: countByChars,
      fadingTier: {
        v: 2,
        budgetTokens: 10_000,
        masked: true,
        latched: true,
      },
    });

    expect(withTier!.remainingContextTokens).toBeGreaterThan(
      withoutTier!.remainingContextTokens!
    );
  });
});

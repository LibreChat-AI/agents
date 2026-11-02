import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import type { TokenCounter } from '@/types';
import { AgentContext } from '../AgentContext';
import { Providers } from '@/common';

describe('AgentContext schema accounting', () => {
  it.each(['tools', 'graphTools'] as const)(
    'counts Zod 4 parameters and descriptions from %s',
    async (field) => {
      const schema = z.object({
        questions: z.array(z.object({ question: z.string() })).min(1),
      });
      const askTool = tool(async () => 'ready', {
        name: 'ask_user_question',
        description: 'Ask related questions and pause the run.',
        schema,
      });
      const context = AgentContext.fromConfig({
        agentId: 'test-agent',
        provider: Providers.OPENAI,
        [field]: [askTool],
      });
      const counted: string[] = [];
      const tokenCounter: TokenCounter = (message) => {
        const content =
          typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content);
        counted.push(content);
        return content.length;
      };

      await context.calculateInstructionTokens(tokenCounter);

      expect(counted).toHaveLength(1);
      expect(JSON.parse(counted[0])).toMatchObject({
        title: 'ask_user_question',
        description: askTool.description,
        properties: {
          questions: {
            type: 'array',
            items: { properties: { question: { type: 'string' } } },
          },
        },
      });
      expect(context.toolSchemaTokens).toBeGreaterThan(0);
      expect(askTool.schema).toBe(schema);
    }
  );
});

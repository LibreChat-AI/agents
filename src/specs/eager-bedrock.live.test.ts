/**
 * Live Bedrock regression coverage for eager event-driven tool execution.
 *
 * Run with:
 * RUN_EAGER_BEDROCK_LIVE=1 npm test -- eager-bedrock.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Run } from '@/run';

const shouldRunLive =
  process.env.RUN_EAGER_BEDROCK_LIVE === '1' &&
  (process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? '') !== '' &&
  (process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ?? '') !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;
const region =
  process.env.BEDROCK_AWS_REGION ??
  process.env.AWS_DEFAULT_REGION ??
  'us-east-1';
const model =
  process.env.BEDROCK_EAGER_LIVE_MODEL ??
  'us.anthropic.claude-sonnet-4-5-20250929-v1:0';

const queryTool: t.LCTool = {
  name: 'run_select_query',
  description: 'Run one read-only ClickHouse SQL query.',
  parameters: {
    type: 'object',
    properties: {
      sql: {
        type: 'string',
        description: 'The complete SQL query to execute.',
      },
    },
    required: ['sql'],
  },
};

function buildPrompt(): string {
  const projections = Array.from(
    { length: 180 },
    (_, index) => `sumIf(value, metric = 'metric_${index}') AS metric_${index}`
  ).join(',\n  ');
  return [
    'Call run_select_query exactly once with the complete SQL below.',
    'Do not shorten, rewrite, or omit any projection.',
    '```sql',
    'SELECT',
    `  ${projections}`,
    'FROM analytics.events',
    'WHERE event_date >= today() - 30 AND tenant_id = \'tenant_live_test\'',
    '```',
  ].join('\n');
}

describeIfLive('Bedrock eager event tool execution (live)', () => {
  jest.setTimeout(180_000);

  it.each([1, 2, 3])(
    'executes large streamed tool call attempt %i once with final arguments',
    async (attempt) => {
      const dispatchedArgs: Array<Record<string, unknown>> = [];
      const completions: t.ToolEndEvent[] = [];
      const agent: t.AgentInputs = {
        agentId: 'bedrock-eager-live',
        provider: Providers.BEDROCK,
        clientOptions: {
          model,
          region,
          credentials: {
            accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
            secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
          },
          temperature: 1,
          maxTokens: 8192,
          streaming: true,
          streamUsage: true,
        },
        instructions:
          'You are a query runner. Always copy user-provided SQL exactly into run_select_query.',
        maxContextTokens: 89000,
        toolDefinitions: [queryTool],
      };

      const run = await Run.create<t.IState>({
        runId: `bedrock-eager-live-${attempt}-${Date.now()}`,
        graphConfig: { type: 'standard', agents: [agent] },
        eagerEventToolExecution: { enabled: true },
        returnContent: true,
        skipCleanup: true,
        customHandlers: {
          [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
          [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
          [GraphEvents.TOOL_END]: new ToolEndHandler(),
          [GraphEvents.ON_TOOL_EXECUTE]: {
            handle: (_event, rawData): void => {
              const data = rawData as t.ToolExecuteBatchRequest;
              dispatchedArgs.push(...data.toolCalls.map((call) => call.args));
              data.resolve(
                data.toolCalls.map((call) => ({
                  toolCallId: call.id,
                  status: 'success' as const,
                  content: 'query completed',
                }))
              );
            },
          },
          [GraphEvents.ON_RUN_STEP_COMPLETED]: {
            handle: (_event, rawData): void => {
              completions.push((rawData as { result: t.ToolEndEvent }).result);
            },
          },
        },
      });

      await run.processStream(
        { messages: [new HumanMessage(buildPrompt())] },
        {
          configurable: {
            provider: Providers.BEDROCK,
            thread_id: `bedrock-eager-live-${attempt}-${Date.now()}`,
          },
          recursionLimit: 8,
          version: 'v2',
        }
      );

      const toolMessages = (run.getRunMessages() ?? []).filter(
        (message) => message.getType() === 'tool'
      );
      const finalToolCall = (run.getRunMessages() ?? [])
        .flatMap((message) =>
          'tool_calls' in message && Array.isArray(message.tool_calls)
            ? message.tool_calls
            : []
        )
        .find((call) => call.name === queryTool.name);
      expect(
        toolMessages.some(
          (message) =>
            typeof message.content === 'string' &&
            message.content.includes('changed after eager execution')
        )
      ).toBe(false);
      expect(dispatchedArgs).toHaveLength(1);
      expect((dispatchedArgs[0].sql as string).length).toBeGreaterThan(8_000);
      expect(finalToolCall?.args).toEqual(dispatchedArgs[0]);
      expect(completions.some((completion) => completion.eager === true)).toBe(
        true
      );
    }
  );
});

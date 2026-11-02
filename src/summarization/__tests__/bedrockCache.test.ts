import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  BedrockRuntimeClient,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  ConverseStreamCommandOutput,
  ConverseStreamOutput,
} from '@aws-sdk/client-bedrock-runtime';
import type { BaseMessage } from '@langchain/core/messages';
import type { TokenCounter } from '@/types';
import { DEFAULT_SUMMARIZATION_PROMPT } from '@/summarization/shared';
import { convertToConverseMessages } from '@/llm/bedrock/utils';
import { addBedrockTailCacheControl } from '@/messages/cache';
import { createSummarizeNode } from '@/summarization/node';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

const count: TokenCounter = (message) =>
  Math.ceil(JSON.stringify(message.content).length / 4) + 4;
const checkpoint = { cachePoint: { type: 'default', ttl: '1h' } };

function createContext(
  model: string,
  promptCache: boolean,
  instructions: string
) {
  return AgentContext.fromConfig(
    {
      agentId: 'agent',
      provider: Providers.BEDROCK,
      instructions,
      clientOptions: {
        model,
        promptCache,
        region: 'us-east-1',
        credentials: {
          accessKeyId: 'offline-test',
          secretAccessKey: 'offline-test',
        },
        _lc_stream_delay: 0,
      },
      maxContextTokens: 32_000,
      summarizationEnabled: true,
      summarizationConfig: {
        maxSummaryTokens: 128,
        retainRecent: { turns: 0 },
      },
    },
    count
  );
}

function summaryResponse(): ConverseStreamCommandOutput {
  return {
    $metadata: {},
    stream: (async function* (): AsyncGenerator<ConverseStreamOutput> {
      yield { messageStart: { role: 'assistant' } };
      yield {
        contentBlockDelta: {
          contentBlockIndex: 0,
          delta: { text: 'CLEAN_SUMMARY' },
        },
      };
      yield { contentBlockStop: { contentBlockIndex: 0 } };
      yield { messageStop: { stopReason: 'end_turn' } };
      yield {
        metadata: {
          usage: { inputTokens: 100, outputTokens: 5, totalTokens: 105 },
          metrics: { latencyMs: 0 },
        },
      };
    })(),
  };
}

afterEach(() => jest.restoreAllMocks());

describe('Bedrock compaction cache prefix', () => {
  it.each([
    ['anthropic.claude-sonnet-5', true, 'The turn system prompt'],
    ['anthropic.claude-sonnet-4-5', true, 'The turn system prompt'],
    ['anthropic.claude-sonnet-5', false, 'The turn system prompt'],
    ['anthropic.claude-sonnet-5', true, ''],
  ] as const)(
    'preserves the turn prefix for %s, cache=%s, system=%s',
    async (model, promptCache, instructions) => {
      const send = jest
        .spyOn(BedrockRuntimeClient.prototype, 'send')
        .mockImplementation(async () => summaryResponse());
      const agentContext = createContext(model, promptCache, instructions);
      await agentContext.tokenCalculationPromise;
      const windowMessages = [
        new HumanMessage('Find the result'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'search', name: 'search', args: {} }],
        }),
        new ToolMessage({ content: 'MASKED_OUTPUT', tool_call_id: 'search' }),
        new AIMessage('The result was found'),
      ];
      const messages = [
        new HumanMessage('OLD_STORED_HISTORY'),
        new AIMessage('Old answer'),
        ...windowMessages,
      ];
      agentContext.pendingOriginalToolContent = new Map([
        [2, 'FULL_STORED_OUTPUT'],
        [4, 'FULL_STORED_OUTPUT'],
      ]);
      const before = JSON.stringify(windowMessages);
      const cachedWindow = promptCache
        ? addBedrockTailCacheControl(windowMessages, '1h')
        : windowMessages;
      const system = agentContext.systemRunnable;
      expect(system != null).toBe(instructions !== '');
      const expectedPrefix =
        system == null ? cachedWindow : await system.invoke(cachedWindow);
      const expectedMessages = [
        ...expectedPrefix,
        new HumanMessage(DEFAULT_SUMMARIZATION_PROMPT),
      ];
      const captured: BaseMessage[][] = [];
      const node = createSummarizeNode({
        agentContext,
        graph: {
          contentData: [],
          contentIndexMap: new Map<string, number>(),
          isMultiAgent: false,
          dispatchRunStep: async () => {},
          dispatchRunStepCompleted: async () => {},
        },
        generateStepId: () => ['summary-step', 0],
      });

      await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            reason: 'manual',
            remainingContextTokens: 0,
            windowMessages,
          },
        },
        {
          callbacks: [
            {
              handleChatModelStart: (_llm, batches) => {
                captured.push(...batches);
              },
            },
          ],
        }
      );

      expect(send).toHaveBeenCalledTimes(1);
      expect(captured).toEqual([expectedMessages]);
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
      const command = send.mock.calls[0][0];
      expect(command).toBeInstanceOf(ConverseStreamCommand);
      if (!(command instanceof ConverseStreamCommand)) {
        throw new Error('Expected a Bedrock ConverseStream request');
      }
      const expected = convertToConverseMessages(expectedMessages, { model });
      expect(command.input.modelId).toBe(model);
      expect(command.input.system).toEqual(expected.converseSystem);
      expect(command.input.messages).toEqual(expected.converseMessages);
      expect(command.input.system).toEqual(
        instructions === ''
          ? []
          : [{ text: instructions }, ...(promptCache ? [checkpoint] : [])]
      );
      expect(command.input.messages?.at(-2)?.content).toEqual([
        { text: 'The result was found' },
        ...(promptCache ? [checkpoint] : []),
      ]);
      expect(command.input.messages?.at(-1)?.content).toEqual([
        { text: DEFAULT_SUMMARIZATION_PROMPT },
      ]);
      const serialized = JSON.stringify(command.input);
      expect(serialized).toContain('MASKED_OUTPUT');
      expect(serialized).not.toContain('FULL_STORED_OUTPUT');
      expect(serialized).not.toContain('OLD_STORED_HISTORY');
      expect(JSON.stringify(windowMessages)).toBe(before);
    }
  );
});

import { RunnableLambda } from '@langchain/core/runnables';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { buildSummaryCarrierText } from '@/summarization/shared';
import { createSummarizeNode } from '@/summarization/node';
import { messagesStateReducer } from '@/messages/reducer';
import { createTokenCounter } from '@/utils/tokens';
import { StandardGraph } from '@/graphs/Graph';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import * as init from '@/llm/init';

const checkpoint = 'Checkpoint of compacted history';
const stableInstructions = 'Stable agent instructions';
const dynamicInstructions = 'Current timestamp: Tuesday';
let tokenCounter: t.TokenCounter;

class SummaryGraph extends StandardGraph {
  dispatchSummaryStep(runStep: t.RunStep, config?: RunnableConfig) {
    return this.trackDispatchedRunStep(runStep, config?.metadata, false);
  }
}

function history(): BaseMessage[] {
  return [1, 2, 3, 4].flatMap((turn) => [
    new HumanMessage({ id: `H${turn}`, content: `H${turn}` }),
    new AIMessage({ id: `A${turn}`, content: `A${turn}` }),
  ]);
}

function cacheAnchors(messages: BaseMessage[]): string[] {
  return messages
    .filter(
      (message) =>
        Array.isArray(message.content) &&
        message.content.some((block) => 'cache_control' in block)
    )
    .map((message) => message.text);
}

function createCompaction(
  provider: Providers.ANTHROPIC | Providers.OPENROUTER,
  overrides: Pick<
    t.AgentInputs,
    'summarizationConfig' | 'additional_instructions' | 'initialSummary'
  > = {}
) {
  const model = new FakeChatModel({ responses: [checkpoint] });
  const stream = jest.spyOn(model, '_streamResponseChunks');
  jest.spyOn(init, 'initializeModel').mockReturnValue(model);
  const graph = new SummaryGraph({
    runId: 'checkpoint-ordering',
    tokenCounter,
    agents: [
      {
        agentId: 'agent',
        provider,
        instructions: stableInstructions,
        clientOptions: { model: 'claude-sonnet-4-5', promptCache: true },
        summarizationEnabled: true,
        ...overrides,
      },
    ],
  });
  const agentContext = graph.agentContexts.get('agent')!;
  const node = RunnableLambda.from(
    createSummarizeNode({
      agentContext,
      graph: {
        contentData: graph.contentData,
        contentIndexMap: graph.contentIndexMap,
        isMultiAgent: false,
        dispatchRunStep: graph.dispatchSummaryStep.bind(graph),
        dispatchRunStepCompleted: (stepId, _result, config) =>
          graph.recordStepCompletion(stepId, { metadata: config?.metadata }),
      },
      generateStepId: graph.generateStepId.bind(graph),
    })
  );
  return { agentContext, node, stream };
}

beforeAll(async () => {
  tokenCounter = await createTokenCounter('claude');
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe.each([Providers.ANTHROPIC, Providers.OPENROUTER] as const)(
  '%s post-compaction checkpoint ordering',
  (provider) => {
    describe.each(['', dynamicInstructions])(
      'dynamic instructions: %j',
      (additionalInstructions) => {
        it.each<{
          label: string;
          summarizationConfig?: t.SummarizationConfig;
          retainedStart: number;
        }>([
          { label: 'default two turns', retainedStart: 4 },
          {
            label: 'explicit one turn',
            summarizationConfig: { retainRecent: { turns: 1 } },
            retainedStart: 6,
          },
          {
            label: 'zero retained turns',
            summarizationConfig: { retainRecent: { turns: 0 } },
            retainedStart: 8,
          },
        ])(
          'places the checkpoint before all retained messages with $label',
          async ({ summarizationConfig, retainedStart }) => {
            const { agentContext, node, stream } = createCompaction(provider, {
              summarizationConfig,
              additional_instructions: additionalInstructions,
            });
            const messages = history();
            const original = JSON.stringify(messages);
            const result = await node.invoke({
              messages,
              summarizationRequest: {
                agentId: 'agent',
                remainingContextTokens: 0,
              },
            });

            expect(stream).toHaveBeenCalledTimes(1);
            expect(result.messages?.[0].getType()).toBe('remove');
            expect(result.messages?.slice(1)).toEqual(
              messages.slice(retainedStart)
            );
            const retained = messagesStateReducer(messages, result.messages!);
            expect(retained).toEqual(messages.slice(retainedStart));
            const projected =
              await agentContext.systemRunnable!.invoke(retained);

            expect(
              projected.map((message) => [message.getType(), message.text])
            ).toEqual([
              ['system', stableInstructions],
              ...(additionalInstructions
                ? [['human', additionalInstructions]]
                : []),
              ['human', buildSummaryCarrierText(checkpoint)],
              ...messages
                .slice(retainedStart)
                .map((message) => [message.getType(), message.text]),
            ]);
            expect(cacheAnchors(projected)).toEqual(
              retained.length > 0
                ? [stableInstructions, 'A4']
                : [stableInstructions]
            );
            expect(JSON.stringify(messages)).toBe(original);
          }
        );
      }
    );

    it('replaces an initial summary and rebuilds the cached prompt before the retained turns', async () => {
      const priorSummary = 'Summary from the previous run';
      const { agentContext, node, stream } = createCompaction(provider, {
        additional_instructions: dynamicInstructions,
        initialSummary: { text: priorSummary, tokenCount: 10 },
      });
      const messages = history();
      const before = await agentContext.systemRunnable!.invoke(messages);
      expect(before.map((message) => message.text)).toContain(
        `${dynamicInstructions}\n\n## Conversation Summary\n\n${priorSummary}`
      );

      const result = await node.invoke({
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
        },
      });

      expect(stream).toHaveBeenCalledTimes(1);
      expect(stream.mock.calls[0][0].at(-1)?.text).toContain(priorSummary);
      expect(agentContext.getSummaryText()).toBe(checkpoint);
      const retained = messagesStateReducer(messages, result.messages!);
      expect(retained).toEqual(messages.slice(4));
      const projected = await agentContext.systemRunnable!.invoke(retained);
      expect(
        projected.map((message) => [message.getType(), message.text])
      ).toEqual([
        ['system', stableInstructions],
        ['human', dynamicInstructions],
        ['human', buildSummaryCarrierText(checkpoint)],
        ['human', 'H3'],
        ['ai', 'A3'],
        ['human', 'H4'],
        ['ai', 'A4'],
      ]);
      expect(cacheAnchors(projected)).toEqual([stableInstructions, 'A4']);
    });
  }
);

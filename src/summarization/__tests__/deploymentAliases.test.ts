import { HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createSummarizeNode } from '@/summarization/node';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

interface SummaryRequest {
  model: string;
  max_tokens?: number;
  max_completion_tokens?: number;
  max_output_tokens?: number;
}

const count: t.TokenCounter = (message) =>
  Math.ceil(JSON.stringify(message.content).length / 4) + 4;

function fixture(provider: Providers.OPENAI | Providers.AZURE) {
  const requests: SummaryRequest[] = [];
  const fetch = jest.fn<
    ReturnType<typeof globalThis.fetch>,
    Parameters<typeof globalThis.fetch>
  >(async (_input, init) => {
    if (typeof init?.body !== 'string') {
      throw new Error('Expected a JSON request body');
    }
    requests.push(JSON.parse(init.body) as SummaryRequest);
    return Response.json({
      id: 'offline-summary',
      object: 'chat.completion',
      created: 0,
      model: 'summary-deployment',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: 'Preserved working context.' },
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    });
  });
  const options: t.OpenAIClientOptions = {
    apiKey: 'offline-test',
    model: 'gpt-6-astra',
    useResponsesApi: false,
    streaming: false,
    disableStreaming: true,
    maxRetries: 0,
    configuration: { fetch, maxRetries: 0 },
  };
  if (provider === Providers.AZURE) {
    Object.assign(options, {
      azureOpenAIApiKey: 'offline-test',
      azureOpenAIApiInstanceName: 'offline-instance',
      azureOpenAIApiDeploymentName: 'summary-deployment',
      azureOpenAIApiVersion: '2024-10-21',
    });
  }
  return { requests, options };
}

async function summarize(
  provider: Providers.OPENAI | Providers.AZURE,
  options: t.OpenAIClientOptions,
  config: t.SummarizationConfig = {}
) {
  const agentContext = AgentContext.fromConfig(
    {
      agentId: 'reasoner',
      provider,
      clientOptions: options,
      maxContextTokens: 100_000,
      summarizationEnabled: true,
      summarizationConfig: {
        prompt: 'Write a checkpoint.',
        retainRecent: { turns: 0 },
        ...config,
      },
    },
    count
  );
  const node = createSummarizeNode({
    agentContext,
    graph: {
      contentData: [] as t.RunStep[],
      contentIndexMap: new Map<string, number>(),
      isMultiAgent: false,
      dispatchRunStep: jest.fn(async (): Promise<void> => {}),
      dispatchRunStepCompleted: jest.fn(async (): Promise<void> => {}),
      closeRunStep: jest.fn(async (): Promise<void> => {}),
    },
    generateStepId: () => ['summary', 0],
  });
  const messages = [new HumanMessage('Keep this context.')];
  await node(
    {
      messages,
      summarizationRequest: {
        agentId: 'reasoner',
        remainingContextTokens: 0,
        reason: 'manual',
        windowMessages: messages,
      },
    },
    {}
  );
  expect(agentContext.getSummaryText()).toBe('Preserved working context.');
}

describe.each([Providers.OPENAI, Providers.AZURE] as const)(
  '%s reasoning summary deployment aliases',
  (provider) => {
    it.each([
      { cap: undefined, parameterCap: undefined, expected: 1536 },
      { cap: 512, parameterCap: undefined, expected: 512 },
      { cap: 512, parameterCap: 256, expected: 256 },
    ])(
      'uses one reasoning token field for %j',
      async ({ cap, parameterCap, expected }) => {
        const { requests, options } = fixture(provider);
        const parameters = {
          modelKwargs: {
            model: 'summary-deployment',
            max_completion_tokens: 1536,
          },
          ...(parameterCap == null ? {} : { maxSummaryTokens: parameterCap }),
        };
        const before = JSON.stringify(parameters);
        await summarize(provider, options, {
          maxSummaryTokens: cap,
          parameters,
          contextBudgetMode: 'window',
        });
        expect(requests).toHaveLength(1);
        expect(requests[0]).toMatchObject({
          model: 'summary-deployment',
          max_completion_tokens: expected,
        });
        expect(requests[0]).not.toHaveProperty('max_tokens');
        expect(requests[0]).not.toHaveProperty('max_output_tokens');
        expect(JSON.stringify(parameters)).toBe(before);
      }
    );

    it('normalizes the fallback allowance through the same transport', async () => {
      const { requests, options } = fixture(provider);
      const primary = {
        ...options,
        configuration: {
          fetch: async () =>
            Response.json(
              { error: { message: 'Unavailable primary' } },
              { status: 400 }
            ),
          maxRetries: 0,
        },
      };
      await summarize(provider, primary, {
        contextBudgetMode: 'window',
        parameters: {
          fallbacks: [
            {
              provider,
              maxContextTokens: 100_000,
              contextBudgetMode: 'window',
              clientOptions: {
                ...options,
                modelKwargs: {
                  model: 'summary-deployment',
                  max_completion_tokens: 1536,
                },
              },
            },
          ],
        },
      });
      expect(requests).toHaveLength(1);
      expect(requests[0]).toMatchObject({
        model: 'summary-deployment',
        max_completion_tokens: 1536,
      });
      expect(requests[0]).not.toHaveProperty('max_tokens');
    });
  }
);

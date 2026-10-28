import { RunnableLambda } from '@langchain/core/runnables';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  DEFAULT_SUMMARIZATION_PROMPT,
  ManualSummarizationSkippedError,
} from '@/summarization/shared';
import { createSummarizeNode } from '@/summarization/node';
import { messagesStateReducer } from '@/messages/reducer';
import { resolveClientOptionsModel } from '@/llm/request';
import { AgentContext } from '@/agents/AgentContext';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import * as init from '@/llm/init';

const count: t.TokenCounter = (message) =>
  Math.ceil(JSON.stringify(message.content).length / 4) + 4;

class CapturingSummaryModel extends FakeChatModel {
  readonly requests: BaseMessage[][] = [];
  failure?: (attempt: number) => Error | undefined;

  constructor() {
    super({ responses: ['PROTECTED_CHECKPOINT'] });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[]
  ): AsyncGenerator<ChatGenerationChunk> {
    this.requests.push([...messages]);
    const error = this.failure?.(this.requests.length);
    if (error != null) throw error;
    yield new ChatGenerationChunk({
      text: 'PROTECTED_CHECKPOINT',
      message: new AIMessageChunk('PROTECTED_CHECKPOINT'),
    });
  }
}

function harness({
  maxContextTokens = 1_800,
  clientOptions,
  summarizationConfig = {
    maxSummaryTokens: 128,
    retainRecent: { turns: 1 },
  },
  model = new CapturingSummaryModel(),
  fallback = new CapturingSummaryModel(),
}: {
  maxContextTokens?: number;
  clientOptions?: t.ClientOptions & Pick<t.BuiltInLLMConfig, 'fallbacks'>;
  summarizationConfig?: t.SummarizationConfig;
  model?: CapturingSummaryModel;
  fallback?: CapturingSummaryModel;
} = {}) {
  const agentContext = AgentContext.fromConfig(
    {
      agentId: 'protected-history',
      provider: Providers.OPENAI,
      instructions: 'Preserve the working context.',
      maxContextTokens,
      clientOptions,
      summarizationEnabled: true,
      summarizationConfig,
    },
    count
  );
  jest
    .spyOn(init, 'initializeModel')
    .mockImplementation(({ clientOptions }) =>
      resolveClientOptionsModel(clientOptions) === 'fallback' ? fallback : model
    );
  const contentData: t.RunStep[] = [];
  const contentIndexMap = new Map<string, number>();
  const completed: t.StepCompleted[] = [];
  const closed: string[] = [];
  const graph = {
    contentData,
    contentIndexMap,
    isMultiAgent: false,
    dispatchRunStep: async (step: t.RunStep): Promise<void> => {
      contentData.push(step);
      contentIndexMap.set(step.id, step.index);
    },
    dispatchRunStepCompleted: async (
      _stepId: string,
      result: t.StepCompleted
    ): Promise<void> => {
      completed.push(result);
    },
    closeRunStep: async (_stepId: string, status: string): Promise<void> => {
      closed.push(status);
    },
  };
  const node = RunnableLambda.from(
    createSummarizeNode({
      agentContext,
      graph,
      generateStepId: () => ['protected-summary-step', 0],
    })
  );
  return { agentContext, model, fallback, graph, completed, closed, node };
}

function history(largeTail = false) {
  const head: BaseMessage[] = [
    new HumanMessage({
      id: 'head-question',
      content: 'head-question ' + 'constraint '.repeat(20),
    }),
    new AIMessage({
      id: 'head-decision',
      content: 'head-decision ' + 'decision '.repeat(20),
    }),
    new HumanMessage({
      id: 'head-followup',
      content: 'head-followup ' + 'requirement '.repeat(20),
    }),
    new AIMessage({
      id: 'head-call',
      content: 'head-call',
      tool_calls: [
        { id: 'head-search', name: 'search', args: { query: 'evidence' } },
      ],
    }),
    new ToolMessage({
      id: 'head-result',
      name: 'search',
      tool_call_id: 'head-search',
      status: 'success',
      content:
        'head-result MASKED_VISIBLE_OUTPUT ' +
        'projected evidence '.repeat(4_000) +
        ' VISIBLE_RESULT_END',
    }),
    new AIMessage({
      id: 'head-outcome',
      content: 'head-outcome Verified evidence.',
    }),
  ];
  const tail: BaseMessage[] = [
    new HumanMessage({
      id: 'tail-question',
      content:
        'tail-question ' + 'recent constraint '.repeat(largeTail ? 1_000 : 10),
    }),
    new AIMessage({
      id: 'tail-call',
      content: [
        {
          type: 'thinking',
          thinking: 'Retain this reasoning.',
          signature: 'tail-call-signature',
        },
        { type: 'text', text: 'tail-call' },
      ],
      tool_calls: [
        { id: 'tail-search', name: 'search', args: { query: 'latest' } },
      ],
    }),
    new ToolMessage({
      id: 'tail-result',
      name: 'search',
      tool_call_id: 'tail-search',
      content: 'tail-result MASKED_RECENT_OUTPUT',
    }),
    new AIMessage({
      id: 'tail-answer',
      content: [
        {
          type: 'thinking',
          thinking: 'Keep the continuation.',
          signature: 'tail-answer-signature',
        },
        { type: 'text', text: 'tail-answer Continue here.' },
      ],
    }),
  ];
  const cold = [
    new HumanMessage({ id: 'cold-question', content: 'COLD_OUTSIDE_WINDOW' }),
    new AIMessage({ id: 'cold-answer', content: 'COLD_ANSWER_OUTSIDE_WINDOW' }),
  ];
  const windowMessages = [...head, ...tail];
  return { head, tail, windowMessages, messages: [...cold, ...windowMessages] };
}

function assertHeadWasSent(head: BaseMessage[], sent: BaseMessage[]): void {
  const headIds = new Set(head.map((message) => message.id));
  expect(
    sent
      .filter((message) => headIds.has(message.id))
      .map((message) => message.id)
  ).toEqual(head.map((message) => message.id));
  for (const original of head) {
    const projected = sent.find((message) => message.id === original.id);
    expect(projected?.getType()).toBe(original.getType());
    expect(JSON.stringify(projected?.content)).toContain(original.id);
    if (original instanceof ToolMessage) {
      expect(projected).toBeInstanceOf(ToolMessage);
      expect(projected).toMatchObject({
        tool_call_id: original.tool_call_id,
        name: original.name,
        status: original.status,
      });
    } else {
      expect(projected?.content).toEqual(original.content);
    }
    if (original instanceof AIMessage) {
      expect(projected).toMatchObject({ tool_calls: original.tool_calls });
    }
  }
}

function seedCheckpoint(agentContext: AgentContext, messages: BaseMessage[]) {
  agentContext.setSummary('PRIOR_PROTECTED_CHECKPOINT', 37, {
    precedesMessages: true,
  });
  const tokenMap: Record<string, number> = {};
  for (let i = 0; i < messages.length; i++) tokenMap[i] = count(messages[i]);
  agentContext.indexTokenCountMap = tokenMap;
  agentContext.baseIndexTokenCountMap = { ...tokenMap };
  return {
    tokenMap,
    baseTokenMap: { ...agentContext.baseIndexTokenCountMap },
    summaryVersion: agentContext.summaryVersion,
    history: JSON.stringify(messages),
  };
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe('protected model-visible compaction history', () => {
  it.each(
    (['manual', 'trigger', 'overflow'] as const).flatMap((reason) =>
      [false, true].map((largeTail) => ({ reason, largeTail }))
    )
  )(
    'covers every removed visible message for $reason with largeTail=$largeTail',
    async ({ reason, largeTail }) => {
      const { agentContext, model, node, completed } = harness();
      await agentContext.tokenCalculationPromise;
      const { messages, windowMessages, head, tail } = history(largeTail);
      const before = JSON.stringify(messages);
      const tailBefore = JSON.stringify(tail);
      agentContext.pendingOriginalToolContent = new Map([
        [6, 'UNMASKED_STORED_SECRET'.repeat(1_000)],
      ]);

      const result = await node.invoke({
        messages,
        summarizationRequest: {
          agentId: agentContext.agentId,
          reason,
          remainingContextTokens: 0,
          allowSummarization: true,
          windowMessages,
        },
      });

      expect(model.requests).toHaveLength(1);
      expect(result.messages?.[0].getType()).toBe('remove');
      const surviving = messagesStateReducer(messages, result.messages ?? []);
      const survivingIds = new Set(surviving.map((message) => message.id));
      const removedVisible = windowMessages.filter(
        (message) => !survivingIds.has(message.id)
      );
      expect(removedVisible).toEqual(head);
      const sent = model.requests[0];
      assertHeadWasSent(removedVisible, sent);
      expect(
        sent.reduce((total, message) => total + count(message), 0)
      ).toBeLessThanOrEqual(1_800);
      expect(sent.at(-1)).toBeInstanceOf(HumanMessage);
      expect(sent.at(-1)?.content).toContain('checkpoint');
      expect(JSON.stringify(sent)).not.toContain('COLD_OUTSIDE_WINDOW');
      expect(JSON.stringify(sent)).not.toContain('UNMASKED_STORED_SECRET');
      const toolResult = sent.find((message) => message.id === 'head-result');
      expect(JSON.stringify(toolResult?.content)).toContain(
        'MASKED_VISIBLE_OUTPUT'
      );
      expect(JSON.stringify(toolResult?.content)).toContain(
        'VISIBLE_RESULT_END'
      );
      expect(JSON.stringify(toolResult?.content).length).toBeLessThan(
        JSON.stringify(head[4].content).length
      );
      expect(
        sent
          .filter((message) => survivingIds.has(message.id))
          .map((message) => message.id)
      ).toEqual(largeTail ? [] : tail.map((message) => message.id));
      expect(surviving).toEqual(tail);
      for (let i = 0; i < tail.length; i++) expect(surviving[i]).toBe(tail[i]);
      expect(JSON.stringify(surviving)).toBe(tailBefore);
      expect(JSON.stringify(messages)).toBe(before);
      expect(agentContext.getSummaryText()).toBe('PROTECTED_CHECKPOINT');
      expect(completed).toHaveLength(1);
    }
  );

  it('keeps the same compactable head through a provider context rejection', async () => {
    const { agentContext, model, node } = harness();
    model.failure = (attempt) =>
      attempt === 1 ? new Error('maximum context length exceeded') : undefined;
    await agentContext.tokenCalculationPromise;
    const { messages, windowMessages, head, tail } = history();
    const result = await node.invoke({
      messages,
      summarizationRequest: {
        agentId: agentContext.agentId,
        reason: 'trigger',
        remainingContextTokens: 0,
        windowMessages,
      },
    });
    expect(model.requests).toHaveLength(2);
    for (const request of model.requests) assertHeadWasSent(head, request);
    expect(
      model.requests[1].reduce((total, message) => total + count(message), 0)
    ).toBeLessThan(
      model.requests[0].reduce((total, message) => total + count(message), 0)
    );
    expect(result.messages?.slice(1)).toEqual(tail);
    expect(agentContext.getSummaryText()).toBe('PROTECTED_CHECKPOINT');
  });

  it.each(
    (['manual', 'trigger', 'overflow'] as const).flatMap((reason) =>
      [false, true].map((withFallback) => ({ reason, withFallback }))
    )
  )(
    'preserves all state when the head cannot fit for $reason with fallback=$withFallback',
    async ({ reason, withFallback }) => {
      const fallbacks: t.FallbackConfig[] = [];
      if (withFallback) {
        fallbacks.push({
          provider: Providers.OPENAI,
          maxContextTokens: 900,
          clientOptions: { model: 'fallback', maxTokens: 64 },
        });
      }
      const { agentContext, model, fallback, node, closed } = harness({
        maxContextTokens: 700,
        clientOptions: { model: 'primary', fallbacks },
      });
      await agentContext.tokenCalculationPromise;
      const { messages, windowMessages, head } = history();
      head[0].content =
        'head-question ' + 'indivisible requirement '.repeat(1_000);
      const before = seedCheckpoint(agentContext, messages);
      const prefixBefore = await agentContext.systemRunnable!.invoke([]);
      const originals = new Map([[6, 'ORIGINAL_STORED_OUTPUT']]);
      agentContext.pendingOriginalToolContent = originals;

      const result = await node.invoke({
        messages,
        summarizationRequest: {
          agentId: agentContext.agentId,
          reason,
          allowSummarization: true,
          remainingContextTokens: 0,
          windowMessages,
        },
      });

      expect(result.messages).toBeUndefined();
      expect(result.manualSummary).toBeUndefined();
      expect(model.requests).toHaveLength(0);
      expect(fallback.requests).toHaveLength(0);
      expect(init.initializeModel).toHaveBeenCalledTimes(withFallback ? 2 : 1);
      expect(JSON.stringify(messages)).toBe(before.history);
      expect(agentContext.getSummaryText()).toBe('PRIOR_PROTECTED_CHECKPOINT');
      expect(agentContext.summaryVersion).toBe(before.summaryVersion);
      expect(agentContext.getTokenBudgetBreakdown().summaryTokens).toBe(37);
      expect(agentContext.hasPendingCompactionSummary()).toBe(true);
      expect(agentContext.indexTokenCountMap).toBe(before.tokenMap);
      expect(agentContext.indexTokenCountMap).toEqual(before.tokenMap);
      expect(agentContext.baseIndexTokenCountMap).toEqual(before.baseTokenMap);
      expect(agentContext.pendingOriginalToolContent).toBe(originals);
      expect(await agentContext.systemRunnable!.invoke([])).toEqual(
        prefixBefore
      );
      expect(closed).toEqual(['failed']);
      agentContext.reset();
      await agentContext.tokenCalculationPromise;
      expect(agentContext.getSummaryText()).toBe('PRIOR_PROTECTED_CHECKPOINT');
      expect(agentContext.baseIndexTokenCountMap).toEqual(before.baseTokenMap);
    }
  );

  it.each(['manual', 'trigger', 'overflow'] as const)(
    'does not summarize a retained-only %s window because cold history exists outside it',
    async (reason) => {
      const { agentContext, model, node, graph } = harness();
      await agentContext.tokenCalculationPromise;
      const { messages, tail } = history();
      const before = seedCheckpoint(agentContext, messages);
      const pending = node.invoke({
        messages,
        summarizationRequest: {
          agentId: agentContext.agentId,
          reason,
          allowSummarization: true,
          remainingContextTokens: 0,
          windowMessages: tail,
        },
      });
      if (reason === 'manual') {
        await expect(pending).rejects.toBeInstanceOf(
          ManualSummarizationSkippedError
        );
        await expect(pending).rejects.toMatchObject({
          reason: 'nothing_to_summarize',
        });
      } else {
        expect((await pending).messages).toBeUndefined();
      }
      expect(model.requests).toHaveLength(0);
      expect(init.initializeModel).not.toHaveBeenCalled();
      expect(graph.contentData).toHaveLength(0);
      expect(JSON.stringify(messages)).toBe(before.history);
      expect(agentContext.getSummaryText()).toBe('PRIOR_PROTECTED_CHECKPOINT');
      expect(agentContext.summaryVersion).toBe(before.summaryVersion);
      expect(agentContext.indexTokenCountMap).toBe(before.tokenMap);
    }
  );

  it('uses the net 129200-token input allowance without reserving agent output twice', async () => {
    const { agentContext, model, node } = harness({
      maxContextTokens: 129_200,
      clientOptions: { model: 'primary', maxTokens: 64_000 },
      summarizationConfig: { retainRecent: { turns: 1 } },
    });
    await agentContext.tokenCalculationPromise;
    const head = new HumanMessage({
      id: 'near-full-head',
      content: 'near-full-head ',
    });
    const tail = [
      new HumanMessage({ id: 'near-full-tail-user', content: 'Continue.' }),
      new AIMessage({ id: 'near-full-tail-ai', content: 'Ready.' }),
    ];
    const windowMessages = [head, ...tail];
    const prefix = await agentContext.systemRunnable!.invoke(windowMessages);
    const initialTokens = prefix.reduce(
      (total, message) => total + count(message),
      count(new HumanMessage(DEFAULT_SUMMARIZATION_PROMPT))
    );
    head.content += 'x'.repeat((129_200 - initialTokens) * 4);
    const before = JSON.stringify(windowMessages);

    const result = await node.invoke({
      messages: windowMessages,
      summarizationRequest: {
        agentId: agentContext.agentId,
        reason: 'manual',
        remainingContextTokens: 0,
        windowMessages,
      },
    });

    expect(model.requests).toHaveLength(1);
    assertHeadWasSent([head], model.requests[0]);
    expect(
      model.requests[0].reduce((total, message) => total + count(message), 0)
    ).toBe(129_200);
    expect(model.requests[0].at(-1)?.content).toBe(
      DEFAULT_SUMMARIZATION_PROMPT
    );
    expect(init.initializeModel).toHaveBeenCalledWith(
      expect.objectContaining({
        clientOptions: expect.objectContaining({ maxTokens: 4_096 }),
      })
    );
    expect(result.messages?.slice(1)).toEqual(tail);
    expect(JSON.stringify(windowMessages)).toBe(before);
    expect(agentContext.getSummaryText()).toBe('PROTECTED_CHECKPOINT');
  });

  it.each([false, true])('keeps corrected input units for overflow with inherited fallback=%s', async withFallback => {
    const model = new CapturingSummaryModel();
    if (withFallback) model.failure = () => new Error('Primary unavailable');
    const {agentContext, node, fallback} = harness({model, maxContextTokens: 32_000,
      clientOptions: {model: 'primary', maxTokens: 4_000, fallbacks: withFallback ? [{provider: Providers.OPENAI,
        clientOptions: {model: 'fallback', maxTokens: 2_660},
      }] : []},
      summarizationConfig: {contextBudgetMode: 'window', prompt: 'Checkpoint. '.repeat(150), maxSummaryTokens: 2_660, retainRecent: {turns: 0}},
    });
    await agentContext.tokenCalculationPromise;
    agentContext.applyContextBudgetCorrection(26_600);
    expect(agentContext.summarizationBudgetMode).toBe('input');
    const head = new HumanMessage({id: 'corrected-head', content: 'x'.repeat(100_000)});
    const result = await node.invoke({messages: [head], summarizationRequest: {agentId: agentContext.agentId,
      remainingContextTokens: 0, reason: 'overflow', allowSummarization: true, windowMessages: [head],
    }});
    const successful = withFallback ? fallback : model;
    expect(successful.requests).toHaveLength(1);
    expect(successful.requests[0]).toContain(head);
    expect(successful.requests[0].reduce((total, message) => total + count(message), 0)).toBeLessThanOrEqual(26_600);
    expect(result.messages?.[0].getType()).toBe('remove');
    expect(agentContext.getSummaryText()).toBe('PROTECTED_CHECKPOINT');
    agentContext.reset();
    expect(agentContext.summarizationBudgetMode).toBe('window');
    expect(agentContext.maxContextTokens).toBe(32_000);
  });

  it.each([
    {
      name: 'explicit input',
      mode: 'input',
      limit: 5_000,
      agentMode: undefined,
      fits: true,
    },
    {
      name: 'explicit window',
      mode: 'window',
      limit: 5_000,
      agentMode: undefined,
      fits: false,
    },
    {
      name: 'default explicit window',
      mode: undefined,
      limit: 5_000,
      agentMode: undefined,
      fits: false,
    },
    {
      name: 'inherited input',
      mode: undefined,
      limit: undefined,
      agentMode: 'input',
      fits: true,
    },
    {
      name: 'inherited window',
      mode: undefined,
      limit: undefined,
      agentMode: 'window',
      fits: false,
    },
  ] as const)(
    'honors the fallback $name budget contract without dropping any head messages',
    async ({ mode, limit, agentMode, fits }) => {
      const model = new CapturingSummaryModel();
      model.failure = () => new Error('Primary unavailable');
      const { agentContext, fallback, node } = harness({
        model,
        maxContextTokens: 5_000,
        clientOptions: {
          model: 'primary',
          fallbacks: [
            {
              provider: Providers.OPENAI,
              maxContextTokens: limit,
              contextBudgetMode: mode,
              clientOptions: { model: 'fallback', maxTokens: 1_000 },
            },
          ],
        },
        summarizationConfig: {
          prompt: 'Checkpoint.',
          updatePrompt: 'Update checkpoint.',
          maxSummaryTokens: 64,
          contextBudgetMode: agentMode,
          retainRecent: { turns: 1 },
        },
      });
      await agentContext.tokenCalculationPromise;
      const head = [
        new HumanMessage({
          id: 'fallback-head',
          content: 'fallback-head ' + 'x'.repeat(18_000),
        }),
      ];
      const tail = [
        new HumanMessage({ id: 'fallback-tail', content: 'Continue.' }),
      ];
      const messages = [...head, ...tail];
      const before = seedCheckpoint(agentContext, messages);
      const result = await node.invoke({
        messages,
        summarizationRequest: {
          agentId: agentContext.agentId,
          reason: 'manual',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      });
      expect(model.requests).toHaveLength(1);
      assertHeadWasSent(head, model.requests[0]);
      if (fits) {
        expect(fallback.requests).toHaveLength(1);
        assertHeadWasSent(head, fallback.requests[0]);
        expect(
          fallback.requests[0].reduce(
            (total, message) => total + count(message),
            0
          )
        ).toBeLessThanOrEqual(5_000);
        expect(result.messages?.slice(1)).toEqual(tail);
        expect(agentContext.getSummaryText()).toBe('PROTECTED_CHECKPOINT');
      } else {
        expect(fallback.requests).toHaveLength(0);
        expect(result.messages).toBeUndefined();
        expect(agentContext.getSummaryText()).toBe(
          'PRIOR_PROTECTED_CHECKPOINT'
        );
        expect(agentContext.summaryVersion).toBe(before.summaryVersion);
        expect(agentContext.indexTokenCountMap).toBe(before.tokenMap);
        expect(agentContext.baseIndexTokenCountMap).toEqual(
          before.baseTokenMap
        );
      }
      expect(JSON.stringify(messages)).toBe(before.history);
    }
  );
});

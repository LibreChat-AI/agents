import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  DEFAULT_SUMMARIZATION_PROMPT,
  DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
  buildSummaryCarrierText,
} from '@/summarization/shared';
import {
  applySummarizationHistoryCache,
  resolveBedrockCompactionCacheModel,
  createSummarizeNode,
} from '@/summarization/node';
import { setFreshProviderMessageProvenance } from '@/messages/provenance';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { createTokenCounter, encodingForModel } from '@/utils/tokens';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { convertInjectedMessages } from '@/messages/injected';
import { Constants, GraphEvents, Providers } from '@/common';
import { registerProvider } from '@/llm/providerRegistry';
import { AgentContext } from '@/agents/AgentContext';
import * as providers from '@/llm/providers';
import * as eventUtils from '@/utils/events';
import { FakeChatModel } from '@/llm/fake';

describe('applySummarizationHistoryCache', () => {
  it('marks Anthropic history before the compaction instruction', () => {
    const cached = applySummarizationHistoryCache({
      messages: [new HumanMessage('history')],
      provider: Providers.ANTHROPIC,
      enabled: true,
    });

    expect(cached[0].content).toEqual([
      {
        type: 'text',
        text: 'history',
        cache_control: { type: 'ephemeral', ttl: '1h' },
      },
    ]);
  });

  it('marks Bedrock history before the compaction instruction', () => {
    const cached = applySummarizationHistoryCache({
      messages: [new HumanMessage('history')],
      provider: Providers.BEDROCK,
      enabled: true,
      bedrockModelId: 'anthropic.claude-sonnet',
    });

    expect(cached[0].content).toEqual([
      { type: 'text', text: 'history' },
      { cachePoint: { type: 'default', ttl: '1h' } },
    ]);
  });

  it('uses five minutes for an explicit non-Claude Bedrock model', () => {
    const cached = applySummarizationHistoryCache({
      messages: [new HumanMessage('history')],
      provider: Providers.BEDROCK,
      enabled: true,
      bedrockModelId: 'amazon.nova-pro-v1:0',
    });

    expect(cached[0].content).toEqual([
      { type: 'text', text: 'history' },
      { cachePoint: { type: 'default' } },
    ]);
  });

  it('keeps the configured model family for an opaque Bedrock profile', () => {
    expect(
      resolveBedrockCompactionCacheModel({
        applicationInferenceProfile:
          'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/opaque',
        model: 'anthropic.claude-sonnet',
      })
    ).toBe('anthropic.claude-sonnet');
  });

  it('does not add provider-specific markers to a fallback provider', () => {
    const messages = [new HumanMessage('history')];

    expect(
      applySummarizationHistoryCache({
        messages,
        provider: Providers.OPENAI,
        enabled: true,
      })
    ).toBe(messages);
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Creates a real AgentContext via fromConfig with sensible defaults.
 *  Extra properties are assigned directly for test-specific overrides.
 *
 *  Defaults `retainRecent.turns` to `0` so that tests which use 1–2 message
 *  states still exercise the LLM-call summarization path.  The recency-window
 *  default of `2` turns would otherwise short-circuit summarization for those
 *  inputs.  Tests that target recency-window behavior should pass an explicit
 *  `summarizationConfig.retainRecent` value. */
function createAgentContext(
  overrides: Record<string, unknown> = {}
): AgentContext {
  const {
    // AgentInputs fields
    agentId = 'agent_0',
    provider = Providers.OPENAI,
    instructions = 'Test agent',
    summarizationEnabled = true,
    summarizationConfig,
    compactionSemanticIndex,
    maxContextTokens,
    tools,
    ...extra
  } = overrides;

  const effectiveSummarizationConfig =
    summarizationConfig != null
      ? summarizationConfig
      : { retainRecent: { turns: 0 } };

  const ctx = AgentContext.fromConfig({
    agentId: agentId as string,
    provider: provider as Providers,
    instructions: instructions as string,
    summarizationEnabled: summarizationEnabled as boolean,
    summarizationConfig: effectiveSummarizationConfig,
    ...(compactionSemanticIndex != null ? { compactionSemanticIndex } : {}),
    ...(maxContextTokens != null ? { maxContextTokens } : {}),
    ...(tools != null ? { tools } : {}),
  } as import('@/types').AgentInputs);

  // Apply direct property overrides for test-specific internal state
  for (const [key, value] of Object.entries(extra)) {
    (ctx as unknown as Record<string, unknown>)[key] = value;
  }

  return ctx;
}

/** Creates a mock graph container for createSummarizeNode. */
function mockGraph(
  onStepCompleted?: (stepId: string, result: t.StepCompleted) => void
): {
  contentData: t.RunStep[];
  contentIndexMap: Map<string, number>;
  config: RunnableConfig;
  runId: string;
  isMultiAgent: boolean;
  dispatchRunStep: (
    runStep: t.RunStep,
    config?: RunnableConfig
  ) => Promise<void>;
  dispatchRunStepCompleted: (
    stepId: string,
    result: t.StepCompleted,
    config?: RunnableConfig
  ) => Promise<void>;
} {
  const contentData: t.RunStep[] = [];
  const contentIndexMap = new Map<string, number>();
  return {
    contentData,
    contentIndexMap,
    config: {} as RunnableConfig,
    runId: 'run_1',
    isMultiAgent: false,
    dispatchRunStep: async (runStep: t.RunStep): Promise<void> => {
      contentData.push(runStep);
      contentIndexMap.set(runStep.id, runStep.index);
    },
    dispatchRunStepCompleted: async (
      stepId: string,
      result: t.StepCompleted
    ): Promise<void> => {
      onStepCompleted?.(stepId, result);
    },
  };
}

let stepCounter = 0;
function generateStepId(_stepKey: string): [string, number] {
  const id = `step_test_${stepCounter++}`;
  return [id, 0];
}

/** Collects custom events dispatched during the node execution. */
function captureEvents(): Array<{ event: string; data: unknown }> {
  const events: Array<{ event: string; data: unknown }> = [];
  jest.spyOn(eventUtils, 'safeDispatchCustomEvent').mockImplementation((async (
    ...args: unknown[]
  ) => {
    events.push({ event: args[0] as string, data: args[1] });
  }) as never);
  return events;
}

/** Creates a mock model that returns a canned response via invoke(). */
function mockInvokeModel(response: string): { invoke: jest.Mock } {
  return {
    invoke: jest.fn().mockResolvedValue({ content: response }),
  };
}

/**
 * Creates a mock model that streams text chunk-by-chunk.
 * invoke() returns the full text; stream() yields one chunk per word.
 */
function mockStreamingModel(response: string): {
  invoke: jest.Mock;
  stream: jest.Mock;
} {
  const words = response.split(' ');
  return {
    invoke: jest.fn().mockResolvedValue({ content: response }),
    stream: jest.fn().mockImplementation(async () => {
      return (async function* (): AsyncGenerator<{ content: string }> {
        for (const word of words) {
          // Add space back except for first word
          yield { content: word + ' ' };
        }
      })();
    }),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  stepCounter = 0;
  jest.restoreAllMocks();
});

describe('createSummarizeNode', () => {
  it('binds the live graph tool projection for cache-aligned compaction', async () => {
    captureEvents();

    const projectedTools = [{ name: 'projected-tool' }] as t.GraphTools;
    const bindTools = jest
      .fn()
      .mockReturnValue(mockInvokeModel('Cache-aligned summary'));
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        bindTools = bindTools;
      } as never
    );

    const agentContext = createAgentContext({
      provider: Providers.ANTHROPIC,
      clientOptions: { promptCache: true },
    });
    const graph = {
      ...mockGraph(),
      getToolsForBinding: jest.fn(() => projectedTools),
    };
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(graph.getToolsForBinding).toHaveBeenCalledWith(
      Providers.ANTHROPIC,
      expect.objectContaining({ promptCache: true })
    );
    expect(bindTools).toHaveBeenCalledWith(projectedTools);
  });

  it('keeps cached raw history identical when semantic guidance is appended', async () => {
    const events = captureEvents();
    const calls: BaseMessage[][] = [];
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        invoke(messages: BaseMessage[]): Promise<{ content: string }> {
          calls.push(messages);
          return Promise.resolve({ content: 'summary' });
        }
      } as never
    );
    const rawHistory = [
      new HumanMessage({ id: 'message-1', content: 'Inspect the runtime' }),
      new AIMessage({ id: 'message-2', content: 'I inspected it' }),
    ];
    setFreshProviderMessageProvenance(rawHistory[0], [
      {
        attribution: 'user',
        sourceMessageId: 'message-1',
        sourceContentPartIndices: [0],
      },
    ]);
    setFreshProviderMessageProvenance(rawHistory[1], [
      {
        attribution: 'model',
        sourceMessageId: 'message-2',
        sourceContentPartIndices: [0],
      },
    ]);
    const run = async (
      compactionSemanticIndex?: t.CompactionSemanticIndex
    ): Promise<void> => {
      const node = createSummarizeNode({
        agentContext: createAgentContext({
          provider: Providers.ANTHROPIC,
          clientOptions: { promptCache: true },
          compactionSemanticIndex,
        }),
        graph: mockGraph(),
        generateStepId,
      });
      await node(
        {
          messages: rawHistory,
          summarizationRequest: {
            remainingContextTokens: 1_000,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      );
    };

    await run();
    await run([
      {
        type: 'activity_phase',
        sourceMessageId: 'message-1',
        sourceContentIndex: 0,
        revision: 1,
        status: 'committed',
        text: 'Mapped the runtime seam',
      },
    ]);

    expect(calls).toHaveLength(2);
    expect(calls[0].slice(0, -1).map((message) => message.toJSON())).toEqual(
      calls[1].slice(0, -1).map((message) => message.toJSON())
    );
    expect(String(calls[0].at(-1)?.content)).not.toContain(
      '<compaction-semantic-index>'
    );
    expect(String(calls[1].at(-1)?.content)).toContain(
      '<compaction-semantic-index>'
    );
    expect(String(calls[1].at(-1)?.content)).toContain(
      DEFAULT_SUMMARIZATION_PROMPT
    );
    const starts = events.filter(
      (event) => event.event === GraphEvents.ON_SUMMARIZE_START
    );
    expect(starts.at(-1)?.data).toMatchObject({
      semanticIndexEntryCount: 1,
      semanticIndexCharCount: expect.any(Number),
    });
  });

  it('emits ON_SUMMARIZE_START and ON_SUMMARIZE_COMPLETE on success', async () => {
    const events = captureEvents();

    // Mock getChatModelClass to return our mock model
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Test summary output');
        }
      } as never
    );

    const agentContext = createAgentContext();
    const graph = mockGraph((_stepId, result) => {
      if (result.type === 'summary') {
        events.push({
          event: GraphEvents.ON_SUMMARIZE_COMPLETE,
          data: { summary: result.summary },
        });
      }
    });
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    const eventNames = events.map((e) => e.event);
    // ON_RUN_STEP now goes through graph.dispatchRunStep, not safeDispatchCustomEvent
    expect(graph.contentData.length).toBeGreaterThan(0);
    expect(eventNames).toContain(GraphEvents.ON_SUMMARIZE_START);
    expect(eventNames).toContain(GraphEvents.ON_SUMMARIZE_COMPLETE);

    // Complete event should have the summary text
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(
      (
        (completeEvent?.data as t.SummarizeCompleteEvent).summary!
          .content?.[0] as { text: string } | undefined
      )?.text
    ).toBe('Test summary output');
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).error
    ).toBeUndefined();
  });

  it('stamps INVOKED_MODEL/INVOKED_PROVIDER metadata for a dedicated summarizer model', async () => {
    captureEvents();

    const capturedConfigs: unknown[] = [];
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(
                async (_messages: unknown, config?: unknown) => {
                  capturedConfigs.push(config);
                  return { content: 'Summary text' };
                }
              ),
          };
        }
      } as never
    );

    const agentContext = createAgentContext({
      summarizationConfig: {
        retainRecent: { turns: 0 },
        model: 'gpt-4.1-mini',
      },
    });
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    /**
     * Usage consumers (the subagent usage-capture handler) attribute the
     * call from these keys — without them, a summarizer model that differs
     * from the agent's primary would be billed against the primary config.
     */
    const config = capturedConfigs[0] as {
      metadata?: Record<string, unknown>;
    };
    expect(config.metadata?.[Constants.INVOKED_MODEL]).toBe('gpt-4.1-mini');
    expect(config.metadata?.[Constants.INVOKED_PROVIDER]).toBe(
      Providers.OPENAI
    );
  });

  it('collects streamed text when model supports stream()', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockStreamingModel('one two three');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test message')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Node collects the full streamed text and calls setSummary.
    // Delta events are dispatched by ChatModelStreamHandler, not the node.
    expect(setSummary).toHaveBeenCalledWith(
      'one two three',
      expect.any(Number)
    );
  });

  it('falls back to invoke when model has no stream()', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Full summary text');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test message')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Falls back to invoke and still collects the text
    expect(setSummary).toHaveBeenCalledWith(
      'Full summary text',
      expect.any(Number)
    );
  });

  it('produces metadata stub when all LLM attempts fail', async () => {
    const events = captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockRejectedValue(new Error('Model error')),
          };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph((_stepId, result) => {
      if (result.type === 'summary') {
        events.push({
          event: GraphEvents.ON_SUMMARIZE_COMPLETE,
          data: { summary: result.summary },
        });
      }
    });
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    const result = await node(
      {
        messages: [new HumanMessage('Test')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(result.summarizationRequest).toBeUndefined();
    // After summarization, REMOVE_ALL + surviving context is returned
    expect(result.messages).toBeDefined();
    expect(result.messages!.length).toBeGreaterThanOrEqual(1);
    expect(result.messages![0]._getType()).toBe('remove');

    // Tier 3 fallback: metadata stub is used as summary text
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(
      (
        (completeEvent?.data as t.SummarizeCompleteEvent).summary!
          .content?.[0] as { text: string } | undefined
      )?.text
    ).toMatch(/^\[Metadata summary:/);
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).error
    ).toBeUndefined();
  });

  it('catches model initialization errors and falls back to metadata stub', async () => {
    captureEvents();

    /**
     * Simulate the "Unsupported LLM provider" case — e.g. when a caller
     * forwards an unrecognized provider name (custom-endpoint label) that
     * getChatModelClass cannot resolve. Prior to the defense-in-depth fix,
     * this error was thrown outside the try/catch in executeSummarizationWithFallback
     * and bubbled up silently. Now it is caught and the metadata stub is used.
     */
    jest.spyOn(providers, 'getChatModelClass').mockImplementation(() => {
      throw new Error('Unsupported LLM provider: Ollama');
    });

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await expect(
      node(
        {
          messages: [new HumanMessage('Test message')],
          summarizationRequest: {
            remainingContextTokens: 1000,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      )
    ).resolves.not.toThrow();

    expect(setSummary).toHaveBeenCalledWith(
      expect.stringContaining('[Metadata summary:'),
      expect.any(Number)
    );
  });

  it('falls back to metadata stub when primary LLM call fails', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockRejectedValue(new Error('LLM unavailable')),
          };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test message')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary).toHaveBeenCalledWith(
      expect.stringContaining('[Metadata summary:'),
      expect.any(Number)
    );
  });

  it('calls setSummary with the final text', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Final summary');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary).toHaveBeenCalledWith(
      'Final summary',
      expect.any(Number)
    );
  });

  it('cache-hit path sends raw messages with instruction appended as final HumanMessage', async () => {
    captureEvents();

    const capturedMessages: Array<{ type: string; content: string }> = [];

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(async (messages: unknown[]) => {
                for (const msg of messages as {
                  getType: () => string;
                  content: string | unknown[];
                }[]) {
                  capturedMessages.push({
                    type: msg.getType(),
                    content:
                      typeof msg.content === 'string'
                        ? msg.content
                        : JSON.stringify(msg.content),
                  });
                }
                return {
                  content:
                    '## Goal\nTest goal\n\n<events>\n<event key="test" turn="0">value</event>\n</events>',
                };
              }),
          };
        }
      } as never
    );

    const agentContext = createAgentContext();
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    await node(
      {
        messages: [
          new HumanMessage('Message 1'),
          new HumanMessage('Message 2'),
          new HumanMessage('Message 3'),
        ],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // The raw messages should be sent + instruction appended as the last HumanMessage
    // messagesToRefine has 3 HumanMessages, instruction adds 1 more
    expect(capturedMessages.length).toBe(4);
    expect(capturedMessages[0].type).toBe('human');
    expect(capturedMessages[0].content).toBe('Message 1');
    expect(capturedMessages[3].type).toBe('human');
    // The last message should contain the summarization prompt
    expect(capturedMessages[3].content).toContain(
      'context window is filling up'
    );
  });

  it('cache-hit path includes prior summary in the instruction message', async () => {
    captureEvents();

    const capturedMessages: Array<{ type: string; content: string }> = [];

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(async (messages: unknown[]) => {
                for (const msg of messages as {
                  getType: () => string;
                  content: string | unknown[];
                }[]) {
                  capturedMessages.push({
                    type: msg.getType(),
                    content:
                      typeof msg.content === 'string'
                        ? msg.content
                        : JSON.stringify(msg.content),
                  });
                }
                return { content: '## Goal\nUpdated summary' };
              }),
          };
        }
      } as never
    );

    // Create context with a prior summary
    const agentContext = createAgentContext();
    agentContext.setSummary('## Goal\nPrior summary content.', 50);

    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('New message')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // The last message should contain the update prompt (prior summary exists)
    const lastMsg = capturedMessages[capturedMessages.length - 1];
    expect(lastMsg.type).toBe('human');
    expect(lastMsg.content).toContain('Merge the new messages');
    // Should include the prior summary
    expect(lastMsg.content).toContain('<previous-summary>');
    expect(lastMsg.content).toContain('Prior summary content');
  });
});

describe('DEFAULT_SUMMARIZATION_PROMPT', () => {
  it('is exported and non-empty', () => {
    expect(typeof DEFAULT_SUMMARIZATION_PROMPT).toBe('string');
    expect(DEFAULT_SUMMARIZATION_PROMPT.length).toBeGreaterThan(0);
  });

  it('contains structured checkpoint sections', () => {
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Goal');
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Progress');
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Key Decisions');
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Next Steps');
  });
});

describe('DEFAULT_UPDATE_SUMMARIZATION_PROMPT', () => {
  it('is exported and non-empty', () => {
    expect(typeof DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toBe('string');
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT.length).toBeGreaterThan(0);
  });

  it('instructs merging new content', () => {
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toMatch(
      /Merge the new messages/i
    );
  });

  it('instructs updating progress tracking', () => {
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toMatch(/Done/);
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toMatch(/In Progress/);
  });
});

describe('budget check — instructions exceed context', () => {
  it('skips summarization when instructionTokens >= maxContextTokens', async () => {
    const events = captureEvents();
    const agentContext = createAgentContext({
      maxContextTokens: 4000,
      systemMessageTokens: 5000,
      formatTokenBudgetBreakdown: () => 'mock breakdown',
    });

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [new HumanMessage('test')],
        summarizationRequest: {
          remainingContextTokens: -1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(result.summarizationRequest).toBeUndefined();
    expect(result.messages).toBeUndefined();

    // No summarization events should have fired
    const summarizeEvents = events.filter(
      (e) =>
        e.event === GraphEvents.ON_SUMMARIZE_START ||
        e.event === GraphEvents.ON_SUMMARIZE_DELTA ||
        e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(summarizeEvents).toHaveLength(0);
  });

  it('proceeds normally when instructionTokens < maxContextTokens', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Budget is fine summary');
        }
      } as never
    );

    const agentContext = createAgentContext({
      maxContextTokens: 8000,
      systemMessageTokens: 2000,
      formatTokenBudgetBreakdown: () => 'mock breakdown',
    });

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [new HumanMessage('hello')],
        summarizationRequest: {
          remainingContextTokens: 500,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Should have summarized — messages returned for state replacement
    expect(result.messages).toBeDefined();
    expect(result.messages!.length).toBeGreaterThan(0);
  });
});

describe('recency window — first-turn protection', () => {
  it('skips the LLM call when only one turn exists (default turns: 2)', async () => {
    const events = captureEvents();

    const invokeMock = jest
      .fn()
      .mockResolvedValue({ content: 'should not be called' });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: {} /* defaults to retainRecent.turns = 2 */,
      setSummary,
    } as never);
    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const largePayload = 'paste'.repeat(10_000);
    const result = await summarizeNode(
      {
        messages: [new HumanMessage(largePayload)],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // No LLM call — first user message is preserved verbatim.
    expect(invokeMock).not.toHaveBeenCalled();
    expect(setSummary).not.toHaveBeenCalled();
    // No state mutation — original messages stay.
    expect(result.messages).toBeUndefined();
    expect(result.summarizationRequest).toBeUndefined();
    // No ON_SUMMARIZE_START emitted on the skip path.
    const eventNames = events.map((e) => e.event);
    expect(eventNames).not.toContain(GraphEvents.ON_SUMMARIZE_START);
    expect(eventNames).not.toContain(GraphEvents.ON_SUMMARIZE_COMPLETE);
  });

  it('skips when a single-turn includes assistant + tool messages', async () => {
    captureEvents();

    const invokeMock = jest.fn().mockResolvedValue({ content: 'unused' });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: {},
      setSummary,
    } as never);
    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [
          new HumanMessage('the first user prompt'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'c', name: 'search', args: {} }],
          }),
          new ToolMessage({
            content: 'result',
            tool_call_id: 'c',
            name: 'search',
          }),
          new AIMessage('here is what i found'),
        ],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(invokeMock).not.toHaveBeenCalled();
    expect(setSummary).not.toHaveBeenCalled();
    expect(result.messages).toBeUndefined();
  });

  it('summarizes older closed tool units inside a long single turn', async () => {
    captureEvents();

    let capturedMessages: BaseMessage[] = [];
    const invokeMock = jest.fn().mockImplementation((messages: unknown) => {
      capturedMessages = messages as BaseMessage[];
      return Promise.resolve({ content: 'Checkpoint of the completed work' });
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: {},
      maxContextTokens: 1_000,
      tokenCounter: () => 100,
      setSummary,
    } as never);
    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });
    const messages: BaseMessage[] = [new HumanMessage('inspect the repo')];
    for (let index = 0; index < 4; index++) {
      const id = `call_${index}`;
      messages.push(
        new AIMessage({
          content: '',
          tool_calls: [{ id, name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: `result ${index}`,
          tool_call_id: id,
          name: 'search',
        })
      );
    }
    messages.push(new AIMessage('preparing the change'));

    const result = await summarizeNode(
      {
        messages,
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(capturedMessages.slice(0, -1)).toEqual(messages.slice(0, 7));
    expect(setSummary).toHaveBeenCalledWith(
      expect.stringContaining('Checkpoint of the completed work'),
      expect.any(Number),
      { precedesMessages: true }
    );
    expect(result.messages?.slice(1)).toEqual(messages.slice(7));
  });

  it('keeps a single-turn history when summarization degrades to metadata', async () => {
    captureEvents();
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: (): never => {
              throw new Error('summarizer unavailable');
            },
            stream: (): never => {
              throw new Error('summarizer unavailable');
            },
          };
        }
      } as never
    );
    const agentContext = createAgentContext({
      summarizationConfig: {},
      maxContextTokens: 1_000,
      tokenCounter: () => 100,
    });
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: mockGraph() as never,
      generateStepId,
    });
    const messages: BaseMessage[] = [new HumanMessage('inspect the repo')];
    for (let index = 0; index < 4; index++) {
      const id = `call_${index}`;
      messages.push(
        new AIMessage({
          content: '',
          tool_calls: [{ id, name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: `result ${index}`,
          tool_call_id: id,
          name: 'search',
        })
      );
    }
    messages.push(new AIMessage('preparing the change'));

    const result = await summarizeNode(
      {
        messages,
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(agentContext.hasSummary()).toBe(false);
    expect(result.messages).toBeUndefined();
  });

  it('still summarizes the head when older turns exist beyond the recency window', async () => {
    captureEvents();

    let capturedMessages: { type: string; content: unknown }[] = [];
    const invokeMock = jest.fn().mockImplementation((messages: unknown) => {
      capturedMessages = (
        messages as Array<{ getType: () => string; content: unknown }>
      ).map((m) => ({ type: m.getType(), content: m.content }));
      return Promise.resolve({ content: 'Summary of older turns' });
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: { retainRecent: { turns: 1 } },
      setSummary,
    } as never);
    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const messages = [
      new HumanMessage('turn 1 query'),
      new AIMessage('turn 1 reply'),
      new HumanMessage('turn 2 query'),
      new AIMessage('turn 2 reply'),
    ];
    const result = await summarizeNode(
      {
        messages,
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Head (turn 1) summarized; tail (turn 2) preserved verbatim.
    expect(setSummary).toHaveBeenCalledWith(
      expect.stringContaining('Summary of older turns'),
      expect.any(Number)
    );
    // Captured messages are the head + the appended summarization instruction.
    // Head has 2 messages (turn 1) + 1 instruction = 3 total.
    expect(capturedMessages).toHaveLength(3);
    expect(capturedMessages[0]?.content).toBe('turn 1 query');
    expect(capturedMessages[1]?.content).toBe('turn 1 reply');

    // Returned messages: removeAll marker + tail.
    expect(result.messages).toBeDefined();
    expect(result.messages![0]?._getType()).toBe('remove');
    expect(result.messages!.slice(1)).toHaveLength(2);
    expect((result.messages![1] as HumanMessage).content).toBe('turn 2 query');
    expect((result.messages![2] as AIMessage).content).toBe('turn 2 reply');
  });

  describe('summary coverage', () => {
    const runCompaction = async (
      messages: BaseMessage[],
      turns = 1
    ): Promise<t.SummaryContentBlock | undefined> => {
      captureEvents();
      jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
        class {
          constructor() {
            return mockInvokeModel('Summary of older turns');
          }
        } as never
      );

      let summaryBlock: t.SummaryContentBlock | undefined;
      const graph = mockGraph((_stepId, result) => {
        if (result.type === 'summary') {
          summaryBlock = result.summary;
        }
      });
      const summarizeNode = createSummarizeNode({
        agentContext: createAgentContext({
          summarizationConfig: { retainRecent: { turns } },
        } as never),
        graph: graph as never,
        generateStepId,
      });

      await summarizeNode(
        {
          messages,
          summarizationRequest: {
            remainingContextTokens: 0,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      );

      return summaryBlock;
    };

    it('records the first retained message as the coverage anchor', async () => {
      const summaryBlock = await runCompaction([
        new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
        new AIMessage({ content: 'turn 1 reply', id: 'm2' }),
        new HumanMessage({ content: 'turn 2 query', id: 'm3' }),
        new AIMessage({ content: 'turn 2 reply', id: 'm4' }),
      ]);

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm3' });
    });

    it('skips a retained message that carries no source id', async () => {
      const summaryBlock = await runCompaction([
        new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
        new AIMessage({ content: 'turn 1 reply', id: 'm2' }),
        new HumanMessage({ content: 'turn 2 query' }),
        new AIMessage({ content: 'turn 2 reply', id: 'm4' }),
      ]);

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm4' });
    });

    it('omits coverage when no retained message carries a source id', async () => {
      const summaryBlock = await runCompaction([
        new HumanMessage('turn 1 query'),
        new AIMessage('turn 1 reply'),
        new HumanMessage('turn 2 query'),
        new AIMessage('turn 2 reply'),
      ]);

      expect(summaryBlock?.coverage).toBeUndefined();
    });

    /** A steer expands one source message into pre-steer, steer, and post-steer
     *  messages sharing its ID, and the recency split lands on the steer. The
     *  straddling message is the anchor, so it survives whole. */
    it('anchors on a source id that straddles the recency boundary', async () => {
      const summaryBlock = await runCompaction([
        new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
        new AIMessage({ content: 'pre-steer reply', id: 'm2' }),
        new HumanMessage({
          content: 'steer',
          id: 'm2',
          additional_kwargs: { role: 'user', source: 'steer' },
        }),
        new AIMessage({ content: 'post-steer reply', id: 'm2' }),
      ]);

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm2' });
    });

    /** A steer carries `source: 'steer'` but is replayed from a payload entry
     *  and stamped with its ID, so it is a valid anchor. When compaction lands
     *  before any post-steer message exists it is the *only* retained entry —
     *  treating every marked message as synthetic drops it. */
    it('anchors on a retained steer with no post-steer message', async () => {
      const summaryBlock = await runCompaction([
        new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
        new AIMessage({ content: 'pre-steer reply', id: 'm2' }),
        new HumanMessage({
          content: 'steer',
          id: 'm2',
          additional_kwargs: { role: 'user', source: 'steer' },
        }),
      ]);

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm2' });
    });

    it('anchors a derived retained message on its persisted source id', async () => {
      const summaryBlock = await runCompaction([
        new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
        new AIMessage({ content: 'pre-steer reply', id: 'm2' }),
        new HumanMessage({
          content: 'steer',
          id: 'reducer-uuid',
          additional_kwargs: {
            role: 'user',
            source: 'steer',
            sourceMessageId: 'm2',
          },
        }),
      ]);

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm2' });
    });

    /** `formatAgentMessages` reconstructs skill bodies inside its payload loop
     *  and keeps processing payload entries after, so this unstamped entry — a
     *  reducer UUID by the time compaction sees it — precedes stamped messages.
     *  Anchoring on it would resolve to nothing on the next run. */
    it('skips a reconstructed skill body to reach the stamped message behind it', async () => {
      const summaryBlock = await runCompaction(
        [
          new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
          new AIMessage({ content: 'turn 1 reply', id: 'm2' }),
          new HumanMessage({
            content: 'skill body',
            id: 'reducer-uuid',
            additional_kwargs: {
              role: 'user',
              isMeta: true,
              source: 'skill',
              skillName: 'demo',
            },
          }),
          new HumanMessage({ content: 'turn 2 query', id: 'm3' }),
          new AIMessage({ content: 'turn 2 reply', id: 'm4' }),
        ],
        2
      );

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm3' });
    });

    /** `InjectedMessage` leaves both `isMeta` and `source` optional, so a bare
     *  injected turn carries no marker of its own — and an injected `steer` is
     *  otherwise indistinguishable from a replayed one. `convertInjectedMessages`
     *  records `injected` on everything it builds, which decides both. */
    it.each([
      ['a bare injected turn', { role: 'user' as const, content: 'injected' }],
      [
        'an injected steer',
        {
          role: 'user' as const,
          content: 'injected steer',
          source: 'steer' as const,
        },
      ],
    ])('skips %s when anchoring', async (_label, injected) => {
      const [converted] = convertInjectedMessages([injected]);
      converted.id = 'reducer-uuid';

      const summaryBlock = await runCompaction(
        [
          new HumanMessage({ content: 'turn 1 query', id: 'm1' }),
          new AIMessage({ content: 'turn 1 reply', id: 'm2' }),
          converted,
          new HumanMessage({ content: 'turn 2 query', id: 'm3' }),
          new AIMessage({ content: 'turn 2 reply', id: 'm4' }),
        ],
        2
      );

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm3' });
    });

    it('anchors on the straddling id when it is the only source', async () => {
      const summaryBlock = await runCompaction([
        new AIMessage({ content: 'pre-steer reply', id: 'm1' }),
        new HumanMessage({
          content: 'steer',
          id: 'm1',
          additional_kwargs: { role: 'user', source: 'steer' },
        }),
        new AIMessage({ content: 'post-steer reply', id: 'm1' }),
      ]);

      expect(summaryBlock?.coverage).toEqual({ retainedFromMessageId: 'm1' });
    });
  });

  it('keeps the masked tail content (does not re-inject restored tool payloads into state)', async () => {
    captureEvents();

    let summarizerSawRestored = false;
    const invokeMock = jest.fn().mockImplementation((messages: unknown) => {
      const arr = messages as Array<{
        getType: () => string;
        content: unknown;
        tool_call_id?: string;
      }>;
      // Confirm the summarizer's input for the restored tool result has
      // the FULL content, not the masked stub.
      summarizerSawRestored = arr.some(
        (m) =>
          m.getType() === 'tool' &&
          typeof m.content === 'string' &&
          (m.content as string).includes('FULL_ORIGINAL_OUTPUT')
      );
      return Promise.resolve({ content: 'summary' });
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: { retainRecent: { turns: 1 } },
      setSummary,
    } as never);
    // Restoration map keyed by message index — applies to head (idx 2)
    // AND to a tool message inside the retained tail (idx 5).  Only the
    // head's restoration should leak into the summarizer; the tail must
    // keep the masked content.
    agentContext.pendingOriginalToolContent = new Map<number, string>([
      [2, 'FULL_ORIGINAL_OUTPUT for head tool result'],
      [5, 'FULL_ORIGINAL_OUTPUT for tail tool result — must NOT survive'],
    ]);

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const headToolCall = new AIMessage({
      content: '',
      tool_calls: [{ id: 'h', name: 'search', args: {} }],
    });
    const headToolResult = new ToolMessage({
      content: 'masked-head-stub',
      tool_call_id: 'h',
      name: 'search',
    });
    const tailToolCall = new AIMessage({
      content: '',
      tool_calls: [{ id: 't', name: 'search', args: {} }],
    });
    const tailToolResult = new ToolMessage({
      content: 'masked-tail-stub',
      tool_call_id: 't',
      name: 'search',
    });
    const messages = [
      new HumanMessage('turn 1 query'),
      headToolCall,
      headToolResult, // index 2 — restored for summarizer
      new HumanMessage('turn 2 query'),
      tailToolCall,
      tailToolResult, // index 5 — must stay masked in returned tail
    ];

    const result = await summarizeNode(
      {
        messages,
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Summarizer saw the full restored head tool result.
    expect(summarizerSawRestored).toBe(true);

    // Returned tail must contain the MASKED tool result, not the restored one.
    expect(result.messages).toBeDefined();
    const tailToolMsg = result.messages!.find(
      (m) => m._getType() === 'tool'
    ) as ToolMessage | undefined;
    expect(tailToolMsg).toBeDefined();
    expect(tailToolMsg!.content).toBe('masked-tail-stub');
    expect(tailToolMsg!.content).not.toContain('FULL_ORIGINAL_OUTPUT');
  });

  it('bounds restored tool originals and preserves ToolMessage metadata', async () => {
    captureEvents();

    let restoredToolMessage: ToolMessage | undefined;
    const invokeMock = jest.fn().mockImplementation((messages: unknown) => {
      restoredToolMessage = (messages as Array<unknown>).find(
        (message) => message instanceof ToolMessage
      ) as ToolMessage | undefined;
      return Promise.resolve({ content: 'summary' });
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const artifact = { source: 'clickhouse', rows: 1_000 };
    const originalToolMessage = new ToolMessage({
      content: 'masked-stub',
      tool_call_id: 'bounded',
      name: 'run_select_query',
      status: 'success',
      artifact,
      metadata: { traceId: 'trace-1' },
      additional_kwargs: { retained: true },
      response_metadata: { requestId: 'request-1' },
    });
    const agentContext = createAgentContext({
      maxContextTokens: 2_000,
      summarizationConfig: { retainRecent: { turns: 0 } },
      setSummary: jest.fn(),
    });
    agentContext.pendingOriginalToolContent = new Map([
      [2, `[{"rows":"${'x'.repeat(20_000)}"}]`],
    ]);

    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: mockGraph() as never,
      generateStepId,
    });
    await summarizeNode(
      {
        messages: [
          new HumanMessage('query the table'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'bounded', name: 'run_select_query', args: {} }],
          }),
          originalToolMessage,
        ],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(restoredToolMessage).toBeDefined();
    expect(typeof restoredToolMessage!.content).toBe('string');
    expect(String(restoredToolMessage!.content).length).toBeLessThanOrEqual(
      2_400
    );
    expect(restoredToolMessage!.content).toContain('[truncated:');
    expect(restoredToolMessage!.status).toBe('success');
    expect(restoredToolMessage!.artifact).toBe(artifact);
    expect(restoredToolMessage!.metadata).toEqual({ traceId: 'trace-1' });
    expect(restoredToolMessage!.additional_kwargs).toEqual({ retained: true });
    expect(restoredToolMessage!.response_metadata).toEqual({
      requestId: 'request-1',
    });
    expect(originalToolMessage.content).toBe('masked-stub');
  });

  it('preserves tail-relevant pendingOriginalToolContent entries (reindexed) for future summaries', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('summary');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: { retainRecent: { turns: 1 } },
      setSummary,
    } as never);
    // Original-content map covers BOTH a head index (1) and tail indices
    // (3, 5).  Only the tail entries should survive, reindexed to the
    // post-removeAll state where tail messages start at 0.
    agentContext.pendingOriginalToolContent = new Map<number, string>([
      [1, 'fullHead'], // belongs to summarized head — should be dropped
      [3, 'fullTailA'], // tail position 3 → reindexed to 0
      [5, 'fullTailB'], // tail position 5 → reindexed to 2
    ]);

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const messages = [
      new HumanMessage('turn 1 query'),
      new AIMessage('turn 1 reply'), // idx 1
      new HumanMessage('turn 2 query'), // idx 2 — tail starts here
      new ToolMessage({
        // idx 3
        content: 'masked-stub-A',
        tool_call_id: 'a',
        name: 'search',
      }),
      new AIMessage('turn 2 reply'), // idx 4
      new ToolMessage({
        // idx 5
        content: 'masked-stub-B',
        tool_call_id: 'b',
        name: 'search',
      }),
    ];

    await summarizeNode(
      {
        messages,
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Summarize fired (we used a real summary text), so head index 1
    // is gone.  Tail entries should remain, reindexed by subtracting
    // tailStartIndex (=2): 3→1, 5→3.
    const carryOver = agentContext.pendingOriginalToolContent;
    expect(carryOver).toBeDefined();
    expect(carryOver!.size).toBe(2);
    expect(carryOver!.get(1)).toBe('fullTailA');
    expect(carryOver!.get(3)).toBe('fullTailB');
    expect(carryOver!.has(0)).toBe(false);
    expect(carryOver!.has(5)).toBe(false);
  });

  it('aligns the dedupe baseline with the surviving tail length after compaction', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('summary');
        }
      } as never
    );

    const agentContext = createAgentContext({
      summarizationConfig: { retainRecent: { turns: 1 } },
    } as never);
    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const messages = [
      new HumanMessage('turn 1 query'),
      new AIMessage('turn 1 reply'),
      new HumanMessage('turn 2 query'),
      new AIMessage('turn 2 reply'),
    ];

    await summarizeNode(
      {
        messages,
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Tail = last turn = 2 messages.  Baseline must equal that count
    // so a follow-up prune call on the unchanged tail short-circuits
    // via shouldSkipSummarization rather than re-triggering compaction.
    expect(agentContext.shouldSkipSummarization(2)).toBe(true);
    expect(agentContext.shouldSkipSummarization(3)).toBe(false);
  });

  it('does not clear pendingOriginalToolContent on the skip path (state unchanged)', async () => {
    captureEvents();

    const invokeMock = jest.fn();
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke: invokeMock };
        }
      } as never
    );

    const agentContext = createAgentContext({
      summarizationConfig: { retainRecent: { turns: 2 } },
    } as never);
    const seededMap = new Map<number, string>([[1, 'preserved-original']]);
    agentContext.pendingOriginalToolContent = seededMap;

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    await summarizeNode(
      {
        messages: [
          new HumanMessage('only turn'),
          new ToolMessage({
            content: 'masked',
            tool_call_id: 'x',
            name: 'search',
          }),
        ],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // No LLM call (skip path); pendingOriginalToolContent must still be
    // available so a future summarization can restore the original.
    expect(invokeMock).not.toHaveBeenCalled();
    expect(agentContext.pendingOriginalToolContent).toBe(seededMap);
    expect(agentContext.pendingOriginalToolContent!.get(1)).toBe(
      'preserved-original'
    );
  });

  it('preserves the legacy "remove all, summary only" shape when retainRecent.turns is 0', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Legacy summary');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({
      summarizationConfig: { retainRecent: { turns: 0 } },
      setSummary,
    } as never);
    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [
          new HumanMessage('only message'),
          new AIMessage('only reply'),
        ],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary).toHaveBeenCalled();
    // Legacy: remove-all only, no tail re-injection.
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0]?._getType()).toBe('remove');
  });
});

describe('emoji-heavy content does not break summarization', () => {
  it('summarization completes without JSON errors on emoji-heavy messages', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Summary of emoji conversation');
        }
      } as never
    );

    const emojiContent = '👨‍💻 coding 🎉 party 🌍 world 🚀 rocket '.repeat(30);
    const agentContext = createAgentContext({
      maxContextTokens: 8000,
      systemMessageTokens: 100,
      formatTokenBudgetBreakdown: () => 'mock breakdown',
    });

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [new HumanMessage(emojiContent)],
        summarizationRequest: {
          remainingContextTokens: 500,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Should complete without throwing JSON serialization errors
    expect(result.messages).toBeDefined();
    expect(result.messages!.length).toBeGreaterThan(0);
  });
});

describe('createSummarizeNode — overflow recovery', () => {
  /**
   * The first recovery compacts deterministically: re-pruning under the
   * corrected budget drives the pruner's tool-output compression. Spending a
   * summarization call there would cost money and risk replacing message
   * content that compression alone can shrink.
   */
  it('skips the model call when summarization is not yet escalated', async () => {
    const modelFactory = jest.spyOn(providers, 'getChatModelClass');
    const agentContext = createAgentContext();
    const graph = mockGraph(() => {});
    const node = createSummarizeNode({ agentContext, graph, generateStepId });

    const result = await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
          reason: 'overflow',
          allowSummarization: false,
        },
      },
      {} as RunnableConfig
    );

    expect(modelFactory).not.toHaveBeenCalled();
    expect(result.summarizationRequest).toBeUndefined();
    /** State untouched, so the agent node re-prunes and retries. */
    expect(result.messages).toBeUndefined();
  });

  it('runs the model call once the recovery escalates', async () => {
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Escalated summary');
        }
      } as never
    );
    const agentContext = createAgentContext();
    const graph = mockGraph(() => {});
    const node = createSummarizeNode({ agentContext, graph, generateStepId });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
          reason: 'overflow',
          allowSummarization: true,
        },
      },
      {} as RunnableConfig
    );

    expect(agentContext.getSummaryText()).toContain('Escalated summary');
  });

  /**
   * The metadata stub describes the history instead of summarizing it, so
   * committing it removes the head and keeps nothing of what it said. That is
   * never an acceptable way to paper over an overflow.
   */
  it('keeps history when the escalated summarizer falls back to a stub', async () => {
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: (): never => {
              throw new Error(
                '400 prompt is too long: 900 tokens > 500 maximum'
              );
            },
            stream: (): never => {
              throw new Error(
                '400 prompt is too long: 900 tokens > 500 maximum'
              );
            },
          };
        }
      } as never
    );
    const agentContext = createAgentContext();
    const graph = mockGraph(() => {});
    const node = createSummarizeNode({ agentContext, graph, generateStepId });

    const result = await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
          reason: 'overflow',
          allowSummarization: true,
        },
      },
      {} as RunnableConfig
    );

    expect(agentContext.hasSummary()).toBe(false);
    expect(result.messages).toBeUndefined();
  });

  it('still commits a stub for a configured trigger, preserving prior behavior', async () => {
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: (): never => {
              throw new Error('upstream unavailable');
            },
            stream: (): never => {
              throw new Error('upstream unavailable');
            },
          };
        }
      } as never
    );
    const agentContext = createAgentContext();
    const graph = mockGraph(() => {});
    const node = createSummarizeNode({ agentContext, graph, generateStepId });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 0,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(agentContext.hasSummary()).toBe(true);
  });
});

describe('createSummarizeNode — empty summarizer output', () => {
  /** Model whose response carries no text — the shape a reasoning-only
   *  completion produces once `extractResponseText` drops thinking blocks. */
  function mockReasoningOnlyModel(): { invoke: jest.Mock } {
    return {
      invoke: jest.fn().mockResolvedValue({
        content: [{ type: 'thinking', thinking: 'deliberating' }],
      }),
    };
  }

  it('leaves the dedupe guard on the current message count', async () => {
    const events = captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockReasoningOnlyModel();
        }
      } as never
    );

    const agentContext = createAgentContext();
    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph() as never,
      generateStepId,
    });

    const messages = [
      new HumanMessage('older question'),
      new AIMessage('older answer'),
      new HumanMessage('latest question'),
    ];

    const result = await node(
      {
        messages,
        summarizationRequest: { remainingContextTokens: 0, agentId: 'agent_0' },
      },
      {} as RunnableConfig
    );

    /** State is untouched: nothing was compacted, so nothing is removed. */
    expect(result.messages).toBeUndefined();
    expect(agentContext.hasSummary()).toBe(false);
    /**
     * The regression: the node used to mark `0` here, which made
     * `shouldSkipSummarization` answer `false` for every later count. The
     * agent node then re-triggered on this same unchanged state, and the run
     * looped through empty summary steps until the graph's recursion cap.
     */
    expect(agentContext.shouldSkipSummarization(messages.length)).toBe(true);
    expect(agentContext.summarizationFailures).toBe(1);

    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect((completeEvent?.data as t.SummarizeCompleteEvent).error).toBe(
      'Summarization produced empty output'
    );
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).summary
    ).toBeUndefined();
  });

  it('stops spending model calls once consecutive empty attempts hit the cap', async () => {
    captureEvents();

    const invoke = jest.fn().mockResolvedValue({ content: '' });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke };
        }
      } as never
    );

    const agentContext = createAgentContext();
    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph() as never,
      generateStepId,
    });

    /** Growing history: each attempt clears the message-count guard on its
     *  own, so only the failure tally can stop the retries. */
    const messages: BaseMessage[] = [];
    for (let attempt = 0; attempt < 6; attempt++) {
      messages.push(new HumanMessage(`question ${attempt}`));
      messages.push(new AIMessage(`answer ${attempt}`));
      await node(
        {
          messages: [...messages],
          summarizationRequest: {
            remainingContextTokens: 0,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      );
    }

    expect(invoke).toHaveBeenCalledTimes(3);
    expect(agentContext.summarizationExhausted).toBe(true);
  });

  it('clears the failure tally once a summary succeeds', async () => {
    captureEvents();

    let call = 0;
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          call++;
          return mockInvokeModel(call === 1 ? '' : 'A real checkpoint');
        }
      } as never
    );

    const agentContext = createAgentContext();
    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph() as never,
      generateStepId,
    });

    const first = [new HumanMessage('q1'), new AIMessage('a1')];
    await node(
      {
        messages: first,
        summarizationRequest: { remainingContextTokens: 0, agentId: 'agent_0' },
      },
      {} as RunnableConfig
    );
    expect(agentContext.summarizationFailures).toBe(1);

    await node(
      {
        messages: [...first, new HumanMessage('q2'), new AIMessage('a2')],
        summarizationRequest: { remainingContextTokens: 0, agentId: 'agent_0' },
      },
      {} as RunnableConfig
    );

    expect(agentContext.summarizationFailures).toBe(0);
    expect(agentContext.summarizationExhausted).toBe(false);
    expect(agentContext.getSummaryText()).toContain('A real checkpoint');
  });

  it('counts a preserved-history stub failure toward the cap', async () => {
    captureEvents();

    const invoke = jest.fn().mockRejectedValue(new Error('provider down'));
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return { invoke };
        }
      } as never
    );

    const agentContext = createAgentContext();
    /** Escalated overflow recovery: the stub is refused so history survives,
     *  which means the attempt made no progress either. */
    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph() as never,
      generateStepId,
    });

    const messages: BaseMessage[] = [];
    for (let attempt = 0; attempt < 5; attempt++) {
      messages.push(new HumanMessage(`question ${attempt}`));
      messages.push(new AIMessage(`answer ${attempt}`));
      await node(
        {
          messages: [...messages],
          summarizationRequest: {
            remainingContextTokens: 0,
            agentId: 'agent_0',
            reason: 'overflow',
            allowSummarization: true,
          },
        },
        {} as RunnableConfig
      );
    }

    expect(invoke).toHaveBeenCalledTimes(3);
    expect(agentContext.hasSummary()).toBe(false);
  });
});

describe('summarize node breaker capture', () => {
  it('stamps the entry-captured breaker epoch into summary attempt metadata', async () => {
    captureEvents();
    const capturedConfigs: Array<{ metadata?: Record<string, unknown> }> = [];
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(
                async (_messages: unknown, config?: unknown) => {
                  capturedConfigs.push(
                    config as { metadata?: Record<string, unknown> }
                  );
                  return { content: 'Summary text' };
                }
              ),
          };
        }
      } as never
    );

    const agentContext = createAgentContext();
    const graph = {
      ...mockGraph(),
      getBreakerEpoch: (): number => 7,
    };
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(capturedConfigs.length).toBeGreaterThan(0);
    for (const config of capturedConfigs) {
      expect(config.metadata?.lc_stream_limit_epoch).toBe(7);
    }
  });

  it('rejects before the model call when the breaker trips during pre-call awaits', async () => {
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });
    const entryBreaker = new AbortController();
    /** The trip lands while ON_SUMMARIZE_START is awaited — after the
     * entry check already passed. */
    jest
      .spyOn(eventUtils, 'safeDispatchCustomEvent')
      .mockImplementation((async (...args: unknown[]) => {
        if (args[0] === GraphEvents.ON_SUMMARIZE_START) {
          entryBreaker.abort(trip);
        }
      }) as never);
    const modelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockReturnValue(
        class {
          constructor() {
            return mockInvokeModel('should never run');
          }
        } as never
      );

    const agentContext = createAgentContext();
    const graph = {
      ...mockGraph(),
      getBreakerSignal: (): AbortSignal => entryBreaker.signal,
    };
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await expect(
      node(
        {
          messages: [new HumanMessage('Hello'), new HumanMessage('World')],
          summarizationRequest: {
            remainingContextTokens: 1000,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      )
    ).rejects.toBe(trip);
    expect(modelClassSpy).not.toHaveBeenCalled();
  });

  it.each([
    { label: 'stream limit', trip: new StreamLimitExceededError({ kind: 'tool_call_args', limit: 10, observed: 11, toolName: 'db_query' }) },
    { label: 'protection', trip: new ProviderTextProtectionError('blocked') },
  ])('rethrows a parent trip on the config signal instead of degrading to the stub ($label)', async ({ trip }) => {
    /** Child graph's own breaker stays live: in a subagent, a ROOT
     * sibling's trip arrives only through the composed invocation signal. */
    const childBreaker = new AbortController();
    const configAbort = new AbortController();
    captureEvents();
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockImplementation(async () => {
              configAbort.abort(trip);
              throw new Error('The operation was aborted');
            }),
          };
        }
      } as never
    );

    const agentContext = createAgentContext();
    const graph = {
      ...mockGraph(),
      getBreakerSignal: (): AbortSignal => childBreaker.signal,
    };
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await expect(
      node(
        {
          messages: [new HumanMessage('Hello'), new HumanMessage('World')],
          summarizationRequest: {
            remainingContextTokens: 1000,
            agentId: 'agent_0',
          },
        },
        { signal: configAbort.signal } as RunnableConfig
      )
    ).rejects.toBe(trip);
  });

  it.each([
    { label: 'stream limit', trip: new StreamLimitExceededError({ kind: 'tool_call_args', limit: 10, observed: 11, toolName: 'db_query' }) },
    { label: 'protection', trip: new ProviderTextProtectionError('blocked') },
  ])('rejects at entry when the breaker has already tripped ($label)', async ({ trip }) => {
    const entryBreaker = new AbortController();
    entryBreaker.abort(trip);

    const modelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockReturnValue(
        class {
          constructor() {
            return mockInvokeModel('should never run');
          }
        } as never
      );

    const agentContext = createAgentContext();
    const graph = {
      ...mockGraph(),
      getBreakerSignal: (): AbortSignal => entryBreaker.signal,
    };
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await expect(
      node(
        {
          messages: [new HumanMessage('Hello'), new HumanMessage('World')],
          summarizationRequest: {
            remainingContextTokens: 1000,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      )
    ).rejects.toBe(trip);

    expect(graph.contentData).toHaveLength(0);
    expect(modelClassSpy).not.toHaveBeenCalled();
  });

  it.each(['reject', 'return'] as const)('does not fallback, complete or commit a summary after an inherited protection trip (%s)', async (mode) => {
    const trip = new ProviderTextProtectionError('blocked');
    const parent = new AbortController();
    const events = captureEvents();
    const invoke = jest.fn(async () => {
      parent.abort(trip);
      if (mode === 'reject') throw new Error('Generic provider abort');
      return new AIMessage('Late summary');
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(class { constructor() { return { invoke }; } } as never);
    const context = createAgentContext();
    const complete = jest.fn();
    const node = createSummarizeNode({ agentContext: context, graph: { ...mockGraph(complete), getBreakerSignal: () => new AbortController().signal }, generateStepId });
    await expect(node({ messages: [new HumanMessage('Hello'), new AIMessage('World')], summarizationRequest: { remainingContextTokens: 1000, agentId: 'agent_0', reason: 'trigger' } }, { signal: parent.signal })).rejects.toBe(trip);
    expect(context.hasSummary()).toBe(false);
    expect(complete).not.toHaveBeenCalled();
    expect(events.some((entry) => entry.event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toBe(false);
    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it('rethrows an inherited protection trip from the fallback instead of committing a periodic stub', async () => {
    const trip = new ProviderTextProtectionError('blocked');
    const parent = new AbortController();
    const events = captureEvents();
    let calls = 0;
    const invoke = jest.fn(async () => {
      if (++calls === 1) throw new Error('Primary unavailable');
      parent.abort(trip);
      throw new Error('Fallback provider aborted');
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(class { constructor() { return { invoke }; } } as never);
    const context = createAgentContext();
    const clientOptions = { model: 'summary-test', ...context.clientOptions, fallbacks: [{ provider: Providers.OPENAI, clientOptions: {} }] };
    context.clientOptions = clientOptions;
    const complete = jest.fn();
    const node = createSummarizeNode({ agentContext: context, graph: { ...mockGraph(complete), getBreakerSignal: () => new AbortController().signal }, generateStepId });
    await expect(node({ messages: [new HumanMessage('Hello'), new AIMessage('World')], summarizationRequest: { remainingContextTokens: 1000, agentId: 'agent_0', reason: 'trigger' } }, { signal: parent.signal })).rejects.toBe(trip);
    expect(calls).toBe(2);
    expect(context.hasSummary()).toBe(false);
    expect(complete).not.toHaveBeenCalled();
    expect(events.some((entry) => entry.event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toBe(false);
  });

  it('binds the model call to the breaker signal read at node entry', async () => {
    const entryBreaker = new AbortController();
    const lateBreaker = new AbortController();
    let started = false;
    jest
      .spyOn(eventUtils, 'safeDispatchCustomEvent')
      .mockImplementation((async (...args: unknown[]) => {
        if (args[0] === GraphEvents.ON_SUMMARIZE_START) {
          started = true;
        }
      }) as never);

    const capturedSignals: Array<AbortSignal | undefined> = [];
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(
                async (_messages: unknown, config?: unknown) => {
                  capturedSignals.push(
                    (config as RunnableConfig | undefined)?.signal
                  );
                  return { content: 'Summary text' };
                }
              ),
          };
        }
      } as never
    );

    const agentContext = createAgentContext();
    /** Simulates a graph reset between node entry and the model call: once
     * ON_SUMMARIZE_START has been awaited, the live accessor hands out a
     * fresh controller's signal. The model call must still see the signal
     * captured at entry. */
    const graph = {
      ...mockGraph(),
      getBreakerSignal: (): AbortSignal =>
        started ? lateBreaker.signal : entryBreaker.signal,
    };
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(capturedSignals.length).toBeGreaterThan(0);
    for (const signal of capturedSignals) {
      expect(signal).toBe(entryBreaker.signal);
    }
  });

  /** `shouldSummarizeOverflow` fires precisely when the host supplied no
   *  tokenCounter, so this branch IS the overflow-recovery summary, and its
   *  count is persisted and then reserved on the retry. It used to be a
   *  four-characters-per-token guess, which understates CJK several-fold and
   *  so under-reserved exactly where a repeat overflow is most likely. Korean
   *  is the sample here because it is the worst measured case. */
  it('measures the carrier with a real tokenizer when the host has none', async () => {
    const summaryBody =
      '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다. 기존 핸들러를 읽어보니 순서가 뒤바뀌어 있었습니다.';
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel(summaryBody);
        }
      } as never
    );

    const agentContext = createAgentContext();
    expect(agentContext.tokenCounter).toBeUndefined();
    const setSummary = jest.spyOn(agentContext, 'setSummary');

    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph(),
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary).toHaveBeenCalled();
    const [persistedText, persistedCount] = setSummary.mock.calls[0] as [
      string,
      number,
    ];
    const carrier = buildSummaryCarrierText(persistedText);
    const counter = await createTokenCounter();

    expect(persistedCount).toBe(counter(new HumanMessage(carrier)));
    /* The guess this replaced. Asserting the gap keeps the test from passing
     * if anyone reintroduces a character heuristic here. */
    expect(persistedCount).toBeGreaterThan(Math.ceil(carrier.length / 4));
  });

  /** The carrier is re-injected into the AGENT's model, so the count has to be
   *  denominated in the agent's tokenizer, not the summarizer's.
   *  `summarizationConfig.model` is undefined for ordinary self-summarization,
   *  so reading the model off the summarizer would silently measure an
   *  Anthropic agent with `o200k_base` and understate it. */
  it('measures with the receiving agent tokenizer, not the summarizer', async () => {
    const summaryBody =
      '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다. 기존 핸들러를 읽어보니 순서가 뒤바뀌어 있었습니다.';
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel(summaryBody);
        }
      } as never
    );

    const agentContext = createAgentContext({
      provider: Providers.ANTHROPIC,
      clientOptions: { model: 'claude-sonnet-4' },
    });
    expect(agentContext.tokenCounter).toBeUndefined();
    /* No dedicated summarizer model: the only model signal is the agent's. */
    expect(agentContext.summarizationConfig?.model).toBeUndefined();
    const setSummary = jest.spyOn(agentContext, 'setSummary');

    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph(),
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    const [persistedText, persistedCount] = setSummary.mock.calls[0] as [
      string,
      number,
    ];
    const carrier = new HumanMessage(buildSummaryCarrierText(persistedText));
    const claudeCounter = await createTokenCounter('claude');
    const openaiCounter = await createTokenCounter('o200k_base');

    expect(persistedCount).toBe(claudeCounter(carrier));
    /* Guards the actual defect: the two encodings must disagree here, or this
     * test would pass just as well while measuring with the wrong one. */
    expect(claudeCounter(carrier)).toBeGreaterThan(openaiCounter(carrier));
  });

  /** `Run.create` derives one counter from `agents[0]` and `StandardGraph`
   *  hands it to every `AgentContext`, so a Claude agent sitting behind a GPT
   *  first agent receives an `o200k_base` counter. Trusting it here measured
   *  the carrier in the wrong units and under-reserved the persisted
   *  checkpoint on the retry, which is the overflow this count exists to
   *  prevent. */
  it('ignores a shared counter built for another agent encoding', async () => {
    const summaryBody =
      '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다. 기존 핸들러를 읽어보니 순서가 뒤바뀌어 있었습니다.';
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel(summaryBody);
        }
      } as never
    );

    const sharedCounter = await createTokenCounter(encodingForModel('gpt-4o'));
    const agentContext = createAgentContext({
      provider: Providers.ANTHROPIC,
      clientOptions: { model: 'claude-sonnet-4' },
      tokenCounter: sharedCounter,
    });
    const setSummary = jest.spyOn(agentContext, 'setSummary');

    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph(),
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    const [persistedText, persistedCount] = setSummary.mock.calls[0] as [
      string,
      number,
    ];
    const carrier = new HumanMessage(buildSummaryCarrierText(persistedText));
    const claudeCounter = await createTokenCounter('claude');

    expect(persistedCount).toBe(claudeCounter(carrier));
    expect(persistedCount).toBeGreaterThan(sharedCounter(carrier));
  });

  /** The other half of the same rule: a counter the host built itself carries
   *  no encoding stamp, and its units are the ones the host's own
   *  `indexTokenCountMap` is denominated in, so it stays authoritative rather
   *  than being second-guessed by a bundled tokenizer. */
  it('keeps using a host-supplied counter of unknown encoding', async () => {
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Checkpoint body');
        }
      } as never
    );

    const hostCounter = jest.fn((): number => 4242);
    const agentContext = createAgentContext({
      provider: Providers.ANTHROPIC,
      clientOptions: { model: 'claude-sonnet-4' },
      tokenCounter: hostCounter,
    });
    const setSummary = jest.spyOn(agentContext, 'setSummary');

    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph(),
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary.mock.calls[0][1]).toBe(4242);
  });

  /** `modelName` is LangChain's alias for `model` and this repository
   *  configures agents through both, so reading only `model` reported an
   *  unconfigured model and fell through to the provider — which for a
   *  Claude-backed OpenRouter agent means `o200k_base` and the same
   *  under-reservation the receiving-agent fix exists to prevent. */
  it('resolves the receiving model through the modelName alias', async () => {
    const summaryBody =
      '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다. 기존 핸들러를 읽어보니 순서가 뒤바뀌어 있었습니다.';
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel(summaryBody);
        }
      } as never
    );

    /* OpenRouter, so the provider fallback cannot stand in for the model: it
     * serves Claude alongside everything else and resolves to `o200k_base`. */
    const agentContext = createAgentContext({
      provider: Providers.OPENROUTER,
      clientOptions: { modelName: 'anthropic/claude-sonnet-4' },
    });
    const setSummary = jest.spyOn(agentContext, 'setSummary');

    const node = createSummarizeNode({
      agentContext,
      graph: mockGraph(),
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    const [persistedText, persistedCount] = setSummary.mock.calls[0] as [
      string,
      number,
    ];
    const carrier = new HumanMessage(buildSummaryCarrierText(persistedText));
    const claudeCounter = await createTokenCounter('claude');
    const openaiCounter = await createTokenCounter('o200k_base');

    expect(persistedCount).toBe(claudeCounter(carrier));
    expect(claudeCounter(carrier)).toBeGreaterThan(openaiCounter(carrier));
  });

  /** A host provider registered with `family: 'anthropic'` serves Claude under
   *  whatever name and deployment alias the host chose, so an exact match on
   *  the `ANTHROPIC` enum missed it and fell back to `o200k_base`. The rest of
   *  the package reads the trait as a family (`isThinkingEnabled`), and the
   *  tokenizer choice now does too. */
  it('honors a registered anthropic family with no configured model', async () => {
    const summaryBody =
      '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다. 기존 핸들러를 읽어보니 순서가 뒤바뀌어 있었습니다.';
    const dispose = registerProvider({
      provider: 'claude-host-summarization-test',
      model: class extends FakeChatModel {
        constructor() {
          super({ responses: [summaryBody] });
        }
      },
      family: 'anthropic',
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel(summaryBody);
        }
      } as never
    );

    try {
      /* No model on either key: the provider is the only signal there is. */
      const agentContext = createAgentContext({
        provider: 'claude-host-summarization-test',
        clientOptions: {},
      });
      const setSummary = jest.spyOn(agentContext, 'setSummary');

      const node = createSummarizeNode({
        agentContext,
        graph: mockGraph(),
        generateStepId,
      });

      await node(
        {
          messages: [new HumanMessage('Hello'), new HumanMessage('World')],
          summarizationRequest: {
            remainingContextTokens: 1000,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      );

      const [persistedText, persistedCount] = setSummary.mock.calls[0] as [
        string,
        number,
      ];
      const carrier = new HumanMessage(buildSummaryCarrierText(persistedText));
      const claudeCounter = await createTokenCounter('claude');
      const openaiCounter = await createTokenCounter('o200k_base');

      expect(persistedCount).toBe(claudeCounter(carrier));
      expect(claudeCounter(carrier)).toBeGreaterThan(openaiCounter(carrier));
    } finally {
      dispose();
    }
  });

  /** `encodingForModel` matches the substring `claude`, so an opaque
   *  deployment alias reports `o200k_base` while proving nothing about the
   *  model behind it. Treating that as the answer skipped the family lookup
   *  entirely and under-reserved the CJK checkpoint by the same 1.9x the
   *  family fix exists to prevent, including against the `o200k_base` counter
   *  `Run.create` stamps from that same alias. */
  it('honors a registered anthropic family behind an opaque model alias', async () => {
    const summaryBody =
      '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다. 기존 핸들러를 읽어보니 순서가 뒤바뀌어 있었습니다.';
    const dispose = registerProvider({
      provider: 'claude-alias-summarization-test',
      model: class extends FakeChatModel {
        constructor() {
          super({ responses: [summaryBody] });
        }
      },
      family: 'anthropic',
    });
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel(summaryBody);
        }
      } as never
    );

    try {
      const sharedCounter = await createTokenCounter(
        encodingForModel('production')
      );
      const agentContext = createAgentContext({
        provider: 'claude-alias-summarization-test',
        clientOptions: { model: 'production' },
        tokenCounter: sharedCounter,
      });
      const setSummary = jest.spyOn(agentContext, 'setSummary');

      const node = createSummarizeNode({
        agentContext,
        graph: mockGraph(),
        generateStepId,
      });

      await node(
        {
          messages: [new HumanMessage('Hello'), new HumanMessage('World')],
          summarizationRequest: {
            remainingContextTokens: 1000,
            agentId: 'agent_0',
          },
        },
        {} as RunnableConfig
      );

      const [persistedText, persistedCount] = setSummary.mock.calls[0] as [
        string,
        number,
      ];
      const carrier = new HumanMessage(buildSummaryCarrierText(persistedText));
      const claudeCounter = await createTokenCounter('claude');

      expect(persistedCount).toBe(claudeCounter(carrier));
      expect(persistedCount).toBeGreaterThan(sharedCounter(carrier));
    } finally {
      dispose();
    }
  });
});

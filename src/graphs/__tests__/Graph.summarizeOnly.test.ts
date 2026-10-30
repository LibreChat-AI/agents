import { MemorySaver } from '@langchain/langgraph';
import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessageChunk,
  HumanMessage,
  AIMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  HookCallback,
  PreCompactHookInput,
  PreCompactHookOutput,
  StopHookOutput,
} from '@/hooks/types';
import type * as t from '@/types';
import { DEFAULT_SUMMARIZATION_PROMPT } from '@/summarization/shared';
import { ManualSummarizationSkippedError } from '@/summarization';
import { HookRegistry } from '@/hooks/HookRegistry';
import { GraphEvents, Providers } from '@/common';
import * as init from '@/llm/init';
import { Run } from '@/run';

const tokenCounter: t.TokenCounter = () => 10;

/** Answers every call with the same text and remembers what it was sent. */
class RecordingModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly calls: BaseMessage[][] = [];

  constructor(
    private readonly reply: string,
    private readonly failure?: Error
  ) {
    super();
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.calls.push(messages);
    if (this.failure != null) {
      throw this.failure;
    }
    return new AIMessageChunk({ content: this.reply });
  }
}

/** A finished conversation: the last message is the assistant's reply. */
function buildHistory(turns: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push(new HumanMessage(`question ${i}`));
    messages.push(new AIMessage(`answer ${i}`));
  }
  return messages;
}

function expectFullHistoryRequest(
  request: BaseMessage[],
  history: BaseMessage[]
): void {
  const contents = (
    messages: BaseMessage[]
  ): Array<{ type: string; content: BaseMessage['content'] }> =>
    messages.map((message) => ({
      type: message.getType(),
      content: message.content,
    }));
  expect(contents(request.slice(0, -1))).toEqual(contents(history));
  expect(request.at(-1)).toBeInstanceOf(HumanMessage);
  expect(request.at(-1)?.content).toBe(DEFAULT_SUMMARIZATION_PROMPT);
}

const streamConfig = {
  configurable: { thread_id: 'summarize-only' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

interface Harness {
  run: Run<t.IState>;
  agentModel: RecordingModel;
  summarizer: RecordingModel;
  snapshots: t.ContextUsageEvent[];
  completions: t.SummarizeCompleteEvent[];
  restore: () => void;
}

interface HarnessOptions {
  runId: string;
  summarizeOnly?: boolean;
  summarizationEnabled?: boolean;
  summarizationConfig?: t.SummarizationConfig;
  withBudget?: boolean;
  summarizerFailure?: Error;
  /** Two agents, `first` and `second`; which one opted in and how they connect. */
  multiAgent?: {
    summarizer: 'first' | 'second';
    edgeType: 'direct' | 'handoff';
  };
  /** Shared across harnesses to replay a checkpointed thread. */
  checkpointer?: MemorySaver;
  hooks?: HookRegistry;
}

async function createHarness(options: HarnessOptions): Promise<Harness> {
  const snapshots: t.ContextUsageEvent[] = [];
  const completions: t.SummarizeCompleteEvent[] = [];
  const withBudget = options.withBudget !== false;
  const llmConfig = {
    provider: Providers.ANTHROPIC,
    disableStreaming: true,
    streamUsage: false,
  };
  const agentFields = {
    ...(withBudget ? { maxContextTokens: 100_000 } : {}),
    summarizationEnabled: options.summarizationEnabled ?? true,
    summarizeOnly: options.summarizeOnly ?? true,
    summarizationConfig: options.summarizationConfig,
  };
  const plainAgent = {
    provider: Providers.ANTHROPIC,
    clientOptions: llmConfig,
    ...(withBudget ? { maxContextTokens: 100_000 } : {}),
  };
  const multi = options.multiAgent;
  const graphConfig: t.RunConfig['graphConfig'] =
    multi != null
      ? {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'first',
            ...plainAgent,
            ...(multi.summarizer === 'first' ? agentFields : {}),
          },
          {
            agentId: 'second',
            ...plainAgent,
            ...(multi.summarizer === 'second' ? agentFields : {}),
          },
        ],
        edges:
            multi.edgeType === 'direct'
              ? [{ from: ['first'], to: ['second'], edgeType: 'direct' }]
              : [{ from: 'first', to: 'second', edgeType: 'handoff' }],
      }
      : {
        type: 'standard',
        llmConfig,
        ...agentFields,
      };
  if (options.checkpointer != null) {
    graphConfig.compileOptions = { checkpointer: options.checkpointer };
  }
  const run = await Run.create<t.IState>({
    runId: options.runId,
    graphConfig,
    hooks: options.hooks,
    returnContent: true,
    skipCleanup: true,
    ...(withBudget ? { tokenCounter } : {}),
    customHandlers: {
      [GraphEvents.ON_CONTEXT_USAGE]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          snapshots.push(data as t.ContextUsageEvent);
        },
      },
      [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          completions.push(data as t.SummarizeCompleteEvent);
        },
      },
    },
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  const agentModel = new RecordingModel('the agent must not answer');
  run.Graph.overrideModel = agentModel;
  const summarizer = new RecordingModel(
    'CHECKPOINT: everything so far',
    options.summarizerFailure
  );
  const spy = jest.spyOn(init, 'initializeModel').mockReturnValue(summarizer);
  return {
    run,
    agentModel,
    summarizer,
    snapshots,
    completions,
    restore: () => spy.mockRestore(),
  };
}

describe('summarize-only runs', () => {
  it('summarizes the full window while retaining two recent turns by default', async () => {
    const harness = await createHarness({ runId: 'summarize-only-basic' });
    try {
      const history = buildHistory(3);
      await harness.run.processStream({ messages: history }, streamConfig);

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(1);
      expectFullHistoryRequest(harness.summarizer.calls[0], history);

      expect(harness.completions).toHaveLength(1);
      expect(harness.completions[0].summary?.content?.[0]).toMatchObject({
        text: 'CHECKPOINT: everything so far',
      });

      /** Exactly one snapshot: the post-summary state the host persists. */
      expect(harness.snapshots).toHaveLength(1);
      expect(harness.snapshots[0].breakdown.summaryTokens).toBeGreaterThan(0);
      expect(harness.snapshots[0].breakdown.messageCount).toBe(4);
    } finally {
      harness.restore();
    }
  });

  it('removes the whole history only when retainRecent.turns is explicitly zero', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-no-retention',
      summarizationConfig: { retainRecent: { turns: 0 } },
    });
    try {
      const history = buildHistory(3);
      await harness.run.processStream({ messages: history }, streamConfig);

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(1);
      expectFullHistoryRequest(harness.summarizer.calls[0], history);
      expect(harness.completions[0].summary).toBeDefined();
      expect(harness.snapshots[0].breakdown.messageCount).toBe(0);
    } finally {
      harness.restore();
    }
  });

  it('keeps an explicitly configured recency window', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-retain',
      summarizationConfig: { retainRecent: { turns: 1 } },
    });
    try {
      const history = buildHistory(3);
      await harness.run.processStream({ messages: history }, streamConfig);

      expect(harness.summarizer.calls).toHaveLength(1);
      expectFullHistoryRequest(harness.summarizer.calls[0], history);
      expect(harness.snapshots).toHaveLength(1);
      expect(harness.snapshots[0].breakdown.messageCount).toBe(2);
    } finally {
      harness.restore();
    }
  });

  it('keeps the default turn window when the host configured only a token cap', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-retain-tokens',
      summarizationConfig: { retainRecent: { tokens: 1_000_000 } },
    });
    try {
      const history = buildHistory(4);
      await harness.run.processStream({ messages: history }, streamConfig);

      expect(harness.summarizer.calls).toHaveLength(1);
      expectFullHistoryRequest(harness.summarizer.calls[0], history);
      expect(harness.snapshots[0].breakdown.messageCount).toBe(4);
    } finally {
      harness.restore();
    }
  });

  it('rejects when the configured window already covers the whole conversation', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-nothing',
      summarizationConfig: { retainRecent: { turns: 5 } },
    });
    try {
      await expect(
        harness.run.processStream({ messages: buildHistory(2) }, streamConfig)
      ).rejects.toMatchObject({
        name: 'ManualSummarizationSkippedError',
        reason: 'nothing_to_summarize',
      });
      expect(harness.summarizer.calls).toHaveLength(0);
      expect(harness.agentModel.calls).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('rejects when summarization is not enabled', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-disabled',
      summarizationEnabled: false,
    });
    try {
      await expect(
        harness.run.processStream({ messages: buildHistory(2) }, streamConfig)
      ).rejects.toBeInstanceOf(ManualSummarizationSkippedError);
      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('keeps the history when every summarizer call fails', async () => {
    /** With no provider able to summarize, the node would otherwise commit
     *  the metadata stub as the checkpoint and remove the whole history. */
    const harness = await createHarness({
      runId: 'summarize-only-provider-failure',
      summarizerFailure: new Error('503 summarizer unavailable'),
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.completions).toHaveLength(1);
      expect(harness.completions[0].summary).toBeUndefined();
      expect(harness.completions[0].error).toMatch(/preserved/);
      expect(
        harness.run.Graph?.agentContexts.get('default')?.getSummaryText()
      ).toBeUndefined();
    } finally {
      harness.restore();
    }
  });

  it('requests the summary even without a pruning budget', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-no-budget',
      withBudget: false,
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(1);
      expect(harness.completions).toHaveLength(1);
      expect(harness.snapshots).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('re-arms for the next run on the same graph', async () => {
    const harness = await createHarness({ runId: 'summarize-only-rearm' });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );
      await harness.run.processStream(
        { messages: buildHistory(4) },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(2);
      expect(harness.completions).toHaveLength(2);
    } finally {
      harness.restore();
    }
  });

  it('runs a summarizer reachable only through a handoff', async () => {
    /** The summarizing agent is not one of the workflow's entry points, and
     *  its predecessor never calls the model to hand off, so the run has to
     *  start at the agent that opted in. */
    const harness = await createHarness({
      runId: 'summarize-only-handoff-target',
      multiAgent: { summarizer: 'second', edgeType: 'handoff' },
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(harness.summarizer.calls).toHaveLength(1);
      expect(harness.completions).toHaveLength(1);
      expect(harness.completions[0].agentId).toBe('second');
      expect(harness.agentModel.calls).toHaveLength(0);
      /** Root trace identity and Langfuse routing follow the run's agent. */
      expect(harness.run.Graph?.defaultAgentId).toBe('second');
    } finally {
      harness.restore();
    }
  });

  it('compiles a direct-edge target to the summarizer alone', async () => {
    /** With the ordinary routing kept, the target would run once from START
     *  and again when its predecessor completed. */
    const harness = await createHarness({
      runId: 'summarize-only-direct-target',
      multiAgent: { summarizer: 'second', edgeType: 'direct' },
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(harness.summarizer.calls).toHaveLength(1);
      expect(harness.completions).toHaveLength(1);
      expect(harness.snapshots).toHaveLength(1);
      expect(harness.agentModel.calls).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('reports a manual trigger and compacted head count to PreCompact hooks', async () => {
    const hooks = new HookRegistry();
    let captured: PreCompactHookInput | undefined;
    const hook: HookCallback<'PreCompact'> = async (
      input
    ): Promise<PreCompactHookOutput> => {
      captured = input;
      return {};
    };
    hooks.register('PreCompact', { hooks: [hook] });
    const harness = await createHarness({
      runId: 'summarize-only-hook',
      hooks,
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(captured?.trigger).toBe('manual');
      expect(captured?.messagesBeforeCount).toBe(2);
      expect(harness.snapshots[0].breakdown.messageCount).toBe(4);
    } finally {
      harness.restore();
    }
  });

  it('clears a checkpointed summary once a later run starts', async () => {
    /** The state channel outlives the run on a checkpointed thread, and a
     *  later ordinary run or a failed compaction must not report it. */
    const checkpointer = new MemorySaver();
    const history = buildHistory(3);
    const retainedContents = history
      .slice(-4)
      .map((message) => message.content);
    const channelsOf = async (): Promise<{
      manualSummary?: string;
      messages?: BaseMessage[];
    }> => {
      const tuple = await checkpointer.getTuple(streamConfig);
      return (tuple?.checkpoint.channel_values ?? {}) as {
        manualSummary?: string;
        messages?: BaseMessage[];
      };
    };
    const summaryOf = async (): Promise<string | undefined> =>
      (await channelsOf()).manualSummary;
    const compaction = await createHarness({
      runId: 'summarize-only-ckpt-a',
      checkpointer,
    });
    try {
      await compaction.run.processStream({ messages: history }, streamConfig);
      const channels = await channelsOf();
      expect(channels.manualSummary).toBe('CHECKPOINT: everything so far');
      /** The outer state is as compacted as the subgraph's: the remove-all
       *  crossed the node boundary instead of merging the tail back. */
      expect(channels.messages?.map((message) => message.content)).toEqual(
        retainedContents
      );
    } finally {
      compaction.restore();
    }

    const ordinary = await createHarness({
      runId: 'summarize-only-ckpt-b',
      summarizeOnly: false,
      checkpointer,
    });
    try {
      await ordinary.run.processStream(
        { messages: [new HumanMessage('a later question')] },
        streamConfig
      );
      expect(ordinary.agentModel.calls).toHaveLength(1);
      expect(
        ordinary.agentModel.calls[0].map((message) => message.content)
      ).toEqual([...retainedContents, 'a later question']);
      expect(await summaryOf()).toBe('');
    } finally {
      ordinary.restore();
    }

    const failing = await createHarness({
      runId: 'summarize-only-ckpt-c',
      checkpointer,
      summarizerFailure: new Error('503 summarizer unavailable'),
    });
    try {
      await failing.run.processStream(
        { messages: buildHistory(1) },
        streamConfig
      );
      expect(failing.completions[0].error).toMatch(/preserved/);
      expect(await summaryOf()).toBe('');
    } finally {
      failing.restore();
    }
  });

  it('admits no stop continuation once the summary exists', async () => {
    /** A blocking Stop hook would otherwise start another graph segment,
     *  which has no claim left to spend and ends without a model call,
     *  until the continuation budget turns the compaction into an error. */
    const hooks = new HookRegistry();
    let stops = 0;
    const hook: HookCallback<'Stop'> = async (): Promise<StopHookOutput> => {
      stops += 1;
      return { decision: 'block', additionalContext: 'Answer the user now.' };
    };
    hooks.register('Stop', { hooks: [hook] });
    const harness = await createHarness({
      runId: 'summarize-only-stop-hook',
      hooks,
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(stops).toBe(1);
      expect(harness.summarizer.calls).toHaveLength(1);
      expect(harness.completions).toHaveLength(1);
      expect(harness.agentModel.calls).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('never lets a chained successor agent call the model', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-successor',
      multiAgent: { summarizer: 'first', edgeType: 'direct' },
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(3) },
        streamConfig
      );

      expect(harness.summarizer.calls).toHaveLength(1);
      expect(harness.completions).toHaveLength(1);
      expect(harness.agentModel.calls).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('leaves ordinary runs on the trigger path', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-off',
      summarizeOnly: false,
    });
    try {
      const content = await harness.run.processStream(
        { messages: [...buildHistory(2), new HumanMessage('one more')] },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(1);
      expect(harness.summarizer.calls).toHaveLength(0);
      expect(content).toEqual([
        { type: 'text', text: 'the agent must not answer' },
      ]);
    } finally {
      harness.restore();
    }
  });
});

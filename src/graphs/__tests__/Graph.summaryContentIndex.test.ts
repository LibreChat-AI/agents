import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessageChunk,
  HumanMessage,
  AIMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import * as init from '@/llm/init';
import { Run } from '@/run';

/**
 * A compaction checkpoint is only usable if the host can read it back. The
 * summarize node builds its own run step, so nothing but the graph's content
 * index decides whether the assistant's own reply lands on the summary's slot
 * and destroys it before the message is saved.
 */

const tokenCounter: t.TokenCounter = (message: BaseMessage) =>
  String(message.content).length + 50;

const streamConfig = {
  configurable: { thread_id: 'summary-content-index' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

/** The compaction request is the only prompt that asks for a summary. */
function isSummarizerCall(messages: BaseMessage[]): boolean {
  return messages.some((message) =>
    String(message.content).includes('<conversation_summary>')
  );
}

class RecordingModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly calls: BaseMessage[][] = [];

  constructor(private readonly reply: string) {
    super();
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.calls.push(messages);
    return new AIMessageChunk({
      content: isSummarizerCall(messages)
        ? 'CHECKPOINT: everything so far'
        : this.reply,
    });
  }
}

function buildHistory(turns: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push(new HumanMessage(`question ${i}`));
    messages.push(new AIMessage(`answer ${i}`));
  }
  return messages;
}

interface Probe {
  steps: t.RunStep[];
  completions: t.SummarizeCompleteEvent[];
  contentParts: Array<t.MessageContentComplex | undefined>;
}

/** Runs one turn through the real graph and the real content aggregator. */
async function runTurn(options: {
  runId: string;
  summarizeOnly: boolean;
  withReply: boolean;
}): Promise<Probe> {
  const steps: t.RunStep[] = [];
  const completions: t.SummarizeCompleteEvent[] = [];
  const { contentParts, aggregateContent } = createContentAggregator();

  const run = await Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.ANTHROPIC,
        disableStreaming: true,
        streamUsage: false,
      },
      maxContextTokens: 1_200,
      summarizationEnabled: true,
      summarizeOnly: options.summarizeOnly,
    },
    returnContent: true,
    skipCleanup: true,
    tokenCounter,
    customHandlers: {
      [GraphEvents.ON_RUN_STEP]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          steps.push({ ...(data as t.RunStep) });
          aggregateContent({
            event: GraphEvents.ON_RUN_STEP,
            data: data as t.RunStep,
          });
        },
      },
      [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          completions.push(data as t.SummarizeCompleteEvent);
          aggregateContent({
            event: GraphEvents.ON_SUMMARIZE_COMPLETE,
            data: data as t.SummarizeCompleteEvent,
          });
        },
      },
    },
  });

  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  const model = new RecordingModel('the agent reply');
  run.Graph.overrideModel = model;
  const spy = jest.spyOn(init, 'initializeModel').mockReturnValue(model);

  const history = buildHistory(12);
  if (options.withReply) {
    history.push(new HumanMessage('what is the total?'));
  }

  try {
    await run.processStream({ messages: history }, streamConfig);
  } finally {
    spy.mockRestore();
  }

  return { steps, completions, contentParts };
}

function summarizeStepIndexes(probe: Probe): number[] {
  const boundaryIds = new Set(
    probe.completions.map(
      (completion) => completion.summary?.boundary?.messageId
    )
  );
  return probe.steps
    .filter((step) => boundaryIds.has(step.id))
    .map((step) => step.index);
}

describe('compaction checkpoint content index', () => {
  it('files the summary on the message the host saves', async () => {
    const probe = await runTurn({
      runId: 'summary-index-summarize-only',
      summarizeOnly: true,
      withReply: false,
    });

    expect(probe.completions).toHaveLength(1);
    expect(
      probe.contentParts.filter((part) => part?.type === 'summary')
    ).toHaveLength(1);
  });

  it('does not let the reply claim the summary content index', async () => {
    const probe = await runTurn({
      runId: 'summary-index-with-reply',
      summarizeOnly: false,
      withReply: true,
    });

    const summaryIndexes = summarizeStepIndexes(probe);
    expect(summaryIndexes).toHaveLength(1);
    /** The checkpoint keeps its slot, and no other step was given the same one. */
    expect(probe.contentParts[summaryIndexes[0]]?.type).toBe('summary');
    expect(
      probe.contentParts.filter((part) => part?.type === 'summary')
    ).toHaveLength(1);
    expect(new Set(probe.steps.map((step) => step.index)).size).toBe(
      probe.steps.length
    );
  });
});

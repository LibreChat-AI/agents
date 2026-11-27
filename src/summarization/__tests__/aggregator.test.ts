import type * as t from '@/types';
import { ContentTypes, GraphEvents, StepTypes } from '@/common';
import { createContentAggregator } from '@/stream';

describe('createContentAggregator – SUMMARY accumulation', () => {
  it('accumulates text from multiple ON_SUMMARIZE_DELTA events', () => {
    const { aggregateContent, contentParts } = createContentAggregator();

    // Register a run step with a summary placeholder
    const runStep: t.RunStep = {
      stepIndex: 0,
      id: 'step_sum_1',
      type: StepTypes.MESSAGE_CREATION,
      index: 0,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'step_sum_1' },
      },
      summary: {
        type: ContentTypes.SUMMARY,
        content: [],
        tokenCount: 0,
        provider: 'openai',
      },
      usage: null,
    };

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: runStep,
    });

    // The run step registration sets the initial placeholder
    expect(contentParts[0]).toEqual(
      expect.objectContaining({ type: ContentTypes.SUMMARY, content: [] })
    );

    // Send multiple deltas with content chunks
    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: {
        id: 'step_sum_1',
        delta: {
          summary: {
            type: ContentTypes.SUMMARY,
            content: [{ type: 'text', text: 'Hello ' }],
            tokenCount: 0,
            provider: 'openai',
          },
        },
      } as t.SummarizeDeltaData,
    });

    const afterFirst = contentParts[0] as t.SummaryContentBlock;
    expect(afterFirst.content).toHaveLength(1);
    expect((afterFirst.content![0] as { text: string }).text).toBe('Hello ');

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: {
        id: 'step_sum_1',
        delta: {
          summary: {
            type: ContentTypes.SUMMARY,
            content: [{ type: 'text', text: 'world!' }],
            tokenCount: 0,
            provider: 'openai',
          },
        },
      } as t.SummarizeDeltaData,
    });

    // Should accumulate content blocks, not replace
    const afterSecond = contentParts[0] as t.SummaryContentBlock;
    expect(afterSecond.content).toHaveLength(2);
    expect((afterSecond.content![0] as { text: string }).text).toBe('Hello ');
    expect((afterSecond.content![1] as { text: string }).text).toBe('world!');
  });

  it('preserves metadata fields from the latest delta', () => {
    const { aggregateContent, contentParts } = createContentAggregator();

    const runStep: t.RunStep = {
      stepIndex: 0,
      id: 'step_sum_2',
      type: StepTypes.MESSAGE_CREATION,
      index: 0,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'step_sum_2' },
      },
      summary: {
        type: ContentTypes.SUMMARY,
        content: [],
        tokenCount: 0,
        provider: 'anthropic',
        model: 'claude-sonnet-4-5',
      },
      usage: null,
    };

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: runStep,
    });

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: {
        id: 'step_sum_2',
        delta: {
          summary: {
            type: ContentTypes.SUMMARY,
            content: [{ type: 'text', text: 'chunk' }],
            tokenCount: 0,
            provider: 'anthropic',
            model: 'claude-sonnet-4-5',
          },
        },
      } as t.SummarizeDeltaData,
    });

    const part = contentParts[0] as t.SummaryContentBlock;
    expect(part.provider).toBe('anthropic');
    expect(part.model).toBe('claude-sonnet-4-5');
  });

  it('preserves agent and parallel-group metadata on summary completion', () => {
    const { aggregateContent, contentParts } = createContentAggregator();

    const runStep: t.RunStep = {
      stepIndex: 0,
      id: 'step_sum_3',
      type: StepTypes.MESSAGE_CREATION,
      index: 0,
      agentId: 'researcher',
      groupId: 7,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'step_sum_3' },
      },
      summary: {
        type: ContentTypes.SUMMARY,
        content: [],
        tokenCount: 0,
        provider: 'openai',
      },
      usage: null,
    };

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: runStep,
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step_sum_3',
          index: 0,
          type: ContentTypes.SUMMARY,
          summary: {
            type: ContentTypes.SUMMARY,
            content: [{ type: ContentTypes.TEXT, text: 'Step summary' }],
            tokenCount: 4,
            provider: 'openai',
          },
        },
      } as unknown as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toEqual(
      expect.objectContaining({
        type: ContentTypes.SUMMARY,
        content: [{ type: ContentTypes.TEXT, text: 'Step summary' }],
        agentId: 'researcher',
        groupId: 7,
      })
    );

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_COMPLETE,
      data: {
        id: 'step_sum_3',
        agentId: 'researcher',
        summary: {
          type: ContentTypes.SUMMARY,
          content: [{ type: ContentTypes.TEXT, text: 'Final summary' }],
          tokenCount: 5,
          provider: 'openai',
          boundary: {
            messageId: 'step_sum_3',
            contentIndex: 0,
          },
        },
      } as t.SummarizeCompleteEvent,
    });

    expect(contentParts[0]).toEqual(
      expect.objectContaining({
        type: ContentTypes.SUMMARY,
        content: [{ type: ContentTypes.TEXT, text: 'Final summary' }],
        tokenCount: 5,
        agentId: 'researcher',
        groupId: 7,
      })
    );
  });

  it('handles delta when no run step exists', () => {
    const { aggregateContent, contentParts } = createContentAggregator();

    // No run step registered — should warn and not crash
    const consoleSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: {
        id: 'nonexistent',
        delta: {
          summary: {
            type: ContentTypes.SUMMARY,
            content: [{ type: 'text', text: 'orphan' }],
            tokenCount: 0,
          },
        },
      } as t.SummarizeDeltaData,
    });

    expect(consoleSpy).toHaveBeenCalled();
    expect(contentParts.length).toBe(0);

    consoleSpy.mockRestore();
  });

  it('discards what was streamed when a delta is marked reset', () => {
    const { aggregateContent, contentParts } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        stepIndex: 0,
        id: 'step_sum_reset',
        type: StepTypes.MESSAGE_CREATION,
        index: 0,
        stepDetails: {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: { message_id: 'step_sum_reset' },
        },
        summary: { type: ContentTypes.SUMMARY, content: [] },
        usage: null,
      } as t.RunStep,
    });
    const delta = (text: string, reset?: boolean): t.SummarizeDeltaData => ({
      id: 'step_sum_reset',
      ...(reset === true ? { reset } : {}),
      delta: {
        summary: {
          type: ContentTypes.SUMMARY,
          content: text === '' ? [] : [{ type: 'text', text }],
          model: 'summarizer',
        },
      },
    });
    const text = (): string =>
      ((contentParts[0] as t.SummaryContentBlock).content ?? [])
        .map((block) => (block as { text: string }).text)
        .join('');

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: delta('abandoned '),
    });
    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: delta('attempt'),
    });
    expect(text()).toBe('abandoned attempt');

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: delta('Second ', true),
    });
    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: delta('attempt'),
    });
    expect(text()).toBe('Second attempt');
    expect(contentParts[0]).toMatchObject({ model: 'summarizer' });

    aggregateContent({
      event: GraphEvents.ON_SUMMARIZE_DELTA,
      data: delta('', true),
    });
    expect(text()).toBe('');
    expect(contentParts[0]).toMatchObject({ type: ContentTypes.SUMMARY });
  });
});

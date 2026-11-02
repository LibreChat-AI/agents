import type * as t from '@/types';
import { GraphEvents, StepTypes, ContentTypes } from '@/common';
import { createContentAggregator } from './stream';

const createRunStep = (id: string): t.RunStep => ({
  id,
  stepIndex: 0,
  type: StepTypes.MESSAGE_CREATION,
  index: 0,
  stepDetails: {
    type: StepTypes.MESSAGE_CREATION,
    message_creation: { message_id: id },
  },
  usage: null,
});

describe('ContentAggregator empty deltas', () => {
  it('should ignore empty message delta content arrays', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: createRunStep('step_empty_message'),
      });

      aggregateContent({
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: {
          id: 'step_empty_message',
          delta: { content: [] },
        },
      });

      expect(warnSpy).not.toHaveBeenCalled();
      expect(contentParts).toEqual([]);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('should ignore empty reasoning delta content arrays', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: createRunStep('step_empty_reasoning'),
      });

      aggregateContent({
        event: GraphEvents.ON_REASONING_DELTA,
        data: {
          id: 'step_empty_reasoning',
          delta: { content: [] },
        },
      });

      expect(warnSpy).not.toHaveBeenCalled();
      expect(contentParts).toEqual([]);
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('ContentAggregator resumed tool completions', () => {
  it('matches a completion to seeded content when a rebuilt run has no step id', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          name: 'approval_probe',
          args: { value: 'original' },
        },
      }
    );

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: '',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_approval',
            name: 'approval_probe',
            args: '{"value":"original"}',
            output: 'approved',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Before approval',
    });
    expect(contentParts[1]).toEqual({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_approval',
        name: 'approval_probe',
        args: '{"value":"original"}',
        type: 'tool_call',
        output: 'approved',
        progress: 1,
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_continuation'),
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_continuation',
        delta: {
          content: [
            {
              type: ContentTypes.TEXT,
              text: 'After approval',
            },
          ],
        },
      } as t.MessageDeltaEvent,
    });

    expect(contentParts[2]).toEqual({
      type: ContentTypes.TEXT,
      text: 'After approval',
    });
  });

  it('rebinds a replacement run step to seeded content by tool call id', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          name: 'approval_probe',
          args: { value: 'original' },
        },
      }
    );

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: {
          id: 'step_rebuilt',
          stepIndex: 0,
          type: StepTypes.TOOL_CALLS,
          index: 0,
          stepDetails: {
            type: StepTypes.TOOL_CALLS,
            tool_calls: [
              {
                id: 'call_approval',
                name: 'approval_probe',
                args: { value: 'original' },
              },
            ],
          },
          usage: null,
        },
      });
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_rebuilt',
            index: 0,
            type: 'tool_call',
            tool_call: {
              id: 'call_approval',
              name: 'approval_probe',
              args: '{"value":"original"}',
              output: 'approved',
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });

      expect(stepMap.get('step_rebuilt')?.index).toBe(1);
      expect(contentParts[0]).toEqual({
        type: ContentTypes.TEXT,
        text: 'Before approval',
      });
      expect(contentParts[1]).toMatchObject({
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          output: 'approved',
          progress: 1,
        },
      });
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: { ...createRunStep('step_continuation'), index: 1 },
      });
      aggregateContent({
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: {
          id: 'step_continuation',
          delta: {
            content: [
              {
                type: ContentTypes.TEXT,
                text: 'After approval',
              },
            ],
          },
        } as t.MessageDeltaEvent,
      });
      expect(stepMap.get('step_continuation')?.index).toBe(2);
      expect(contentParts[2]).toEqual({
        type: ContentTypes.TEXT,
        text: 'After approval',
      });
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('matches parallel completions to their own seeded tool cards', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_first',
          name: 'approval_probe',
          args: { value: 'first' },
        },
      },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_second',
          name: 'approval_probe',
          args: { value: 'second' },
        },
      }
    );

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_parallel',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_first',
              name: 'approval_probe',
              args: { value: 'first' },
            },
            {
              id: 'call_second',
              name: 'approval_probe',
              args: { value: 'second' },
            },
          ],
        },
        usage: null,
      },
    });

    for (const [id, value] of [
      ['call_first', 'first'],
      ['call_second', 'second'],
    ]) {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_parallel',
            index: 0,
            type: 'tool_call',
            tool_call: {
              id,
              name: 'approval_probe',
              args: JSON.stringify({ value }),
              output: `${value} approved`,
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });
    }

    expect(contentParts[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_first',
        output: 'first approved',
        progress: 1,
      },
    });
    expect(contentParts[2]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_second',
        output: 'second approved',
        progress: 1,
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: { ...createRunStep('step_after_parallel'), index: 1 },
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_after_parallel',
        delta: {
          content: [
            {
              type: ContentTypes.TEXT,
              text: 'Parallel tools complete',
            },
          ],
        },
      } as t.MessageDeltaEvent,
    });
    expect(contentParts[3]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Parallel tools complete',
    });
  });

  it('appends unmatched tools and continuation content after the entire seed', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          name: 'approval_probe',
          args: {},
        },
      },
      { type: ContentTypes.TEXT, text: 'Seeded tail one' },
      { type: ContentTypes.TEXT, text: 'Seeded tail two' }
    );

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_rebuilt',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_approval',
              name: 'approval_probe',
              args: {},
            },
            {
              id: 'call_new',
              name: 'approval_probe',
              args: { value: 'new' },
            },
          ],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: { ...createRunStep('step_after_seed'), index: 1 },
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_after_seed',
        delta: {
          content: [{ type: ContentTypes.TEXT, text: 'After seed' }],
        },
      } as t.MessageDeltaEvent,
    });

    expect(stepMap.get('step_rebuilt')?.index).toBe(1);
    expect(stepMap.get('step_after_seed')?.index).toBe(5);
    expect(contentParts.slice(2, 4)).toEqual([
      { type: ContentTypes.TEXT, text: 'Seeded tail one' },
      { type: ContentTypes.TEXT, text: 'Seeded tail two' },
    ]);
    expect(contentParts[4]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_new',
        args: { value: 'new' },
      },
    });
    expect(contentParts[5]).toEqual({
      type: ContentTypes.TEXT,
      text: 'After seed',
    });
  });

  it('preserves seeded metadata when a blank-id completion replaces a tool card', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push({
      type: ContentTypes.TOOL_CALL,
      agentId: 'researcher',
      groupId: 7,
      tool_call: {
        id: 'call_with_metadata',
        name: 'approval_probe',
        args: {},
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: '',
          index: 42,
          type: 'tool_call',
          tool_call: {
            id: 'call_with_metadata',
            name: 'approval_probe',
            args: '{}',
            output: 'approved',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toMatchObject({
      agentId: 'researcher',
      groupId: 7,
      tool_call: {
        id: 'call_with_metadata',
        output: 'approved',
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_after_large_completion_index'),
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_after_large_completion_index',
        delta: {
          content: [{ type: ContentTypes.TEXT, text: 'Still dense' }],
        },
      } as t.MessageDeltaEvent,
    });
    expect(stepMap.get('step_after_large_completion_index')?.index).toBe(1);
    expect(contentParts[1]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Still dense',
    });
  });

  it('applies metadata from a hydrated run step to its seeded tool card', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_hydrated',
        name: 'approval_probe',
        args: {},
      },
    });
    stepMap.set('step_hydrated', {
      id: 'step_hydrated',
      stepIndex: 0,
      type: StepTypes.TOOL_CALLS,
      index: 0,
      agentId: 'reviewer',
      groupId: 11,
      stepDetails: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          {
            id: 'call_hydrated',
            name: 'approval_probe',
            args: {},
          },
        ],
      },
      usage: null,
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step_hydrated',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_hydrated',
            name: 'approval_probe',
            args: '{}',
            output: 'approved',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toMatchObject({
      agentId: 'reviewer',
      groupId: 11,
      tool_call: { output: 'approved' },
    });
  });

  it('drops an unknown completion for a parallel step', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();
    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: {
          id: 'step_parallel_unknown',
          stepIndex: 0,
          type: StepTypes.TOOL_CALLS,
          index: 0,
          stepDetails: {
            type: StepTypes.TOOL_CALLS,
            tool_calls: [
              { id: 'call_known_first', name: 'first', args: {} },
              { id: 'call_known_second', name: 'second', args: {} },
            ],
          },
          usage: null,
        },
      });
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_parallel_unknown',
            index: 999,
            type: 'tool_call',
            tool_call: {
              id: 'call_unknown',
              name: 'unknown',
              args: '{}',
              output: 'must not overwrite',
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });

      expect(contentParts).toHaveLength(2);
      expect(contentParts[0]?.type).toBe(ContentTypes.TOOL_CALL);
      expect(contentParts[1]?.type).toBe(ContentTypes.TOOL_CALL);
      expect(warnSpy).toHaveBeenCalledWith(
        'No run step or tool call found for completed step event'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('drops a completion that matches neither a step nor seeded tool call', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: '',
            index: 0,
            type: 'tool_call',
            tool_call: {
              id: 'call_unknown',
              name: 'unknown',
              args: '{}',
              output: 'untrusted',
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });

      expect(contentParts).toEqual([]);
      expect(warnSpy).toHaveBeenCalledWith(
        'No run step or tool call found for completed step event'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('ContentAggregator physical content indices', () => {
  it('allocates physical slots independently from streamed chunk keys', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_streamed_tool',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_streamed_tool',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              index: 2,
              id: 'call_streamed_first',
              name: 'approval_probe',
              args: '{"value":"first"}',
            },
            {
              index: 5,
              id: 'call_streamed_second',
              name: 'approval_probe',
              args: '{"value":"second"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(2);
    expect(contentParts[0]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_streamed_first',
        name: 'approval_probe',
      },
    });
    expect(contentParts[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_streamed_second',
        name: 'approval_probe',
      },
    });

    for (const [id, output] of [
      ['call_streamed_first', 'first complete'],
      ['call_streamed_second', 'second complete'],
    ]) {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_streamed_tool',
            index: 500,
            type: 'tool_call',
            tool_call: {
              id,
              name: 'approval_probe',
              args: '{}',
              output,
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });
    }
    expect(contentParts[0]).toMatchObject({
      tool_call: { output: 'first complete' },
    });
    expect(contentParts[1]).toMatchObject({
      tool_call: { output: 'second complete' },
    });
  });

  it('keeps distinct chunk keys separate when tool ids arrive late', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_late_ids',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_late_ids',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [{ index: 2, name: 'first', args: '{"value":"first"}' }],
        },
      } as t.RunStepDeltaEvent,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_late_ids',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 5, name: 'second', args: '{"value":"second"}' },
          ],
        },
      } as t.RunStepDeltaEvent,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_late_ids',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 2, id: 'call_late_first', name: 'first', args: {} },
            { index: 5, id: 'call_late_second', name: 'second', args: {} },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(2);
    expect(contentParts[0]).toMatchObject({
      tool_call: {
        id: 'call_late_first',
        name: 'first',
      },
    });
    expect(contentParts[1]).toMatchObject({
      tool_call: {
        id: 'call_late_second',
        name: 'second',
      },
    });
  });

  it('uses the reserved slot when a streamed tool id has no chunk index', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_without_chunk_index',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_without_chunk_index',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_without_chunk_index',
              name: 'approval_probe',
              args: '{}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(1);
    expect(contentParts[0]).toMatchObject({
      tool_call: {
        id: 'call_without_chunk_index',
        name: 'approval_probe',
      },
    });
  });

  it('routes an id-less chunk to the only known tool card', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_single_tool',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_single',
              name: 'approval_probe',
              args: {},
            },
          ],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_single_tool',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              index: 9,
              name: 'approval_probe',
              args: '{"value":"updated"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(1);
    expect(contentParts[0]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_single',
        args: '{"value":"updated"}',
      },
    });
  });

  it('routes indexed id-less chunks across known parallel tool cards', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_parallel_chunks',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { id: 'call_chunk_first', name: 'first', args: {} },
            { id: 'call_chunk_second', name: 'second', args: {} },
          ],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_parallel_chunks',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 7, name: 'first', args: '{"value":"first"}' },
            { index: 9, name: 'second', args: '{"value":"second"}' },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts[0]).toMatchObject({
      tool_call: {
        id: 'call_chunk_first',
        args: '{"value":"first"}',
      },
    });
    expect(contentParts[1]).toMatchObject({
      tool_call: {
        id: 'call_chunk_second',
        args: '{"value":"second"}',
      },
    });
  });

  it('places a raw agent update after seeded content', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Seed one' },
      { type: ContentTypes.TEXT, text: 'Seed two' }
    );

    aggregateContent({
      event: GraphEvents.ON_AGENT_UPDATE,
      data: {
        type: ContentTypes.AGENT_UPDATE,
        agent_update: {
          index: 0,
          runId: 'run_resumed',
          agentId: 'researcher',
        },
      },
    });

    expect(contentParts[2]).toEqual({
      type: ContentTypes.AGENT_UPDATE,
      agent_update: {
        index: 2,
        runId: 'run_resumed',
        agentId: 'researcher',
      },
    });
  });
});

describe('ContentAggregator provider-specific parts', () => {
  it('should preserve Gemini server-side tool content blocks', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };
    const toolResponsePart: t.MessageContentComplex = {
      type: 'toolResponse',
      toolResponse: {
        id: 'server-search-1',
        name: 'google_search',
        response: { results: [] },
      },
    };

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_tool_call'),
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_tool_call',
        delta: { content: [toolCallPart] },
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: { ...createRunStep('step_tool_response'), index: 1 },
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_tool_response',
        delta: { content: [toolResponsePart] },
      },
    });

    expect(contentParts[0]).toEqual(toolCallPart);
    expect(contentParts[1]).toEqual(toolResponsePart);
  });
});

describe('ContentAggregator multi-entry deltas', () => {
  it('concatenates every text entry of a message delta in order', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_multi_text'),
    });

    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_multi_text',
        delta: {
          content: [
            { type: ContentTypes.TEXT, text: 'Hello ' },
            { type: ContentTypes.TEXT, text: 'streaming ' },
            { type: ContentTypes.TEXT, text: 'world' },
          ],
        },
      },
    });

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Hello streaming world',
    });
  });

  it('concatenates every think entry of a reasoning delta in order', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_multi_think'),
    });

    aggregateContent({
      event: GraphEvents.ON_REASONING_DELTA,
      data: {
        id: 'step_multi_think',
        delta: {
          content: [
            { type: ContentTypes.THINK, think: 'First reasoning block. ' },
            { type: ContentTypes.THINK, think: 'Second reasoning block.' },
          ],
        },
      },
    });

    expect(contentParts[0]).toEqual({
      type: ContentTypes.THINK,
      think: 'First reasoning block. Second reasoning block.',
    });
  });

  it('accumulates across multi-entry and single-entry deltas alike', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_mixed_cadence'),
    });

    aggregateContent({
      event: GraphEvents.ON_REASONING_DELTA,
      data: {
        id: 'step_mixed_cadence',
        delta: {
          content: [
            { type: ContentTypes.THINK, think: 'One. ' },
            { type: ContentTypes.THINK, think: 'Two. ' },
          ],
        },
      },
    });

    aggregateContent({
      event: GraphEvents.ON_REASONING_DELTA,
      data: {
        id: 'step_mixed_cadence',
        delta: { content: [{ type: ContentTypes.THINK, think: 'Three.' }] },
      },
    });

    expect(contentParts[0]).toEqual({
      type: ContentTypes.THINK,
      think: 'One. Two. Three.',
    });
  });
});

describe('ContentAggregator citation deltas', () => {
  const citation = (url: string, cited_text: string) => ({
    type: 'web_search_result_location',
    url,
    title: 'Example',
    cited_text,
    encrypted_index: 'enc',
  });

  /**
   * Anthropic emits a search turn's citations as their own `citations_delta`,
   * which `_makeMessageChunkFromAnthropicEvent` normalizes into a `text` part
   * carrying `citations` and **no** `text` key.
   */
  const citationDelta = (id: string, ...citations: unknown[]) => ({
    event: GraphEvents.ON_MESSAGE_DELTA,
    data: {
      id,
      delta: { content: [{ type: ContentTypes.TEXT, citations }] },
    } as t.MessageDeltaEvent,
  });

  const textDelta = (id: string, text: string) => ({
    event: GraphEvents.ON_MESSAGE_DELTA,
    data: {
      id,
      delta: { content: [{ type: ContentTypes.TEXT, text }] },
    } as t.MessageDeltaEvent,
  });

  it('keeps citations that arrive without accompanying text', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_citations'),
    });

    aggregateContent(textDelta('step_citations', 'Answer.'));
    aggregateContent(
      citationDelta('step_citations', citation('https://a.example', 'quoted a'))
    );

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Answer.',
      citations: [citation('https://a.example', 'quoted a')],
    });
  });

  it('accumulates citations across successive deltas', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_multi'),
    });

    aggregateContent(textDelta('step_multi', 'Part one. '));
    aggregateContent(
      citationDelta('step_multi', citation('https://a.example', 'quoted a'))
    );
    aggregateContent(textDelta('step_multi', 'Part two.'));
    aggregateContent(
      citationDelta('step_multi', citation('https://b.example', 'quoted b'))
    );

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Part one. Part two.',
      citations: [
        citation('https://a.example', 'quoted a'),
        citation('https://b.example', 'quoted b'),
      ],
    });
  });

  it('leaves parts without citations untouched', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_plain'),
    });

    aggregateContent(textDelta('step_plain', 'Hello '));
    aggregateContent(textDelta('step_plain', 'world.'));

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Hello world.',
    });
    expect(contentParts[0]).not.toHaveProperty('citations');
  });

  it('preserves tool_call_ids when a citations-only delta follows', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_tool_ids'),
    });

    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_tool_ids',
        delta: {
          content: [
            {
              type: ContentTypes.TEXT,
              text: 'Answer.',
              tool_call_ids: ['call_1'],
            },
          ],
        },
      } as t.MessageDeltaEvent,
    });
    aggregateContent(
      citationDelta('step_tool_ids', citation('https://a.example', 'quoted a'))
    );

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Answer.',
      tool_call_ids: ['call_1'],
      citations: [citation('https://a.example', 'quoted a')],
    });
  });
});

describe('ContentAggregator dispatched tool calls', () => {
  it('keeps the dispatched target on the completed dispatcher call', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_late',
        name: 'call_tool',
        args: '{"name":"late_tool","arguments":"{}"}',
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: '',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_late',
            name: 'call_tool',
            args: '{"name":"late_tool","arguments":"{}"}',
            output: 'found',
            progress: 1,
            dispatched: { name: 'late_tool', args: '{}' },
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_late',
        name: 'call_tool',
        args: '{"name":"late_tool","arguments":"{}"}',
        type: 'tool_call',
        output: 'found',
        progress: 1,
        dispatched: { name: 'late_tool', args: '{}' },
      },
    });
  });
});

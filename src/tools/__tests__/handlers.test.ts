import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  handleToolCallChunks,
  handleToolCalls,
  handleServerToolResult,
} from '../handlers';
import { StepTypes, ToolCallTypes, Providers, GraphEvents } from '@/common';

type MockGraph = {
  getStepKey: jest.Mock;
  getStepIdByKey: jest.Mock;
  getRunStep: jest.Mock;
  dispatchRunStep: jest.Mock;
  dispatchRunStepDelta: jest.Mock;
  registerPendingToolCall: jest.Mock;
  toolCallStepIds: Map<string, string>;
  messageStepHasToolCalls: Map<string, boolean>;
  messageIdsByStepKey: Map<string, string>;
  prelimMessageIdsByStepKey: Map<string, string>;
  invokedToolIds?: Set<string>;
  handlerRegistry?: {
    getHandler: jest.Mock;
  };
};

function createMockGraph(overrides?: Partial<MockGraph>): MockGraph {
  let stepCounter = 0;
  return {
    getStepKey: jest.fn<() => string>().mockReturnValue('step-key'),
    getStepIdByKey: jest.fn<() => string>().mockReturnValue('prev-step-id'),
    getRunStep: jest
      .fn<() => t.RunStep | undefined>()
      .mockReturnValue(undefined),
    dispatchRunStep: jest
      .fn<() => Promise<string>>()
      .mockImplementation(async () => `new-step-${++stepCounter}`),
    dispatchRunStepDelta: jest
      .fn<() => Promise<void>>()
      .mockResolvedValue(undefined),
    registerPendingToolCall: jest.fn<() => void>(),
    toolCallStepIds: new Map(),
    messageStepHasToolCalls: new Map(),
    messageIdsByStepKey: new Map(),
    prelimMessageIdsByStepKey: new Map(),
    invokedToolIds: undefined,
    handlerRegistry: undefined,
    ...overrides,
  };
}

function makeRunStep(
  type: StepTypes,
  opts?: { tool_calls?: t.AgentToolCall[]; id?: string; index?: number }
): t.RunStep {
  const stepDetails: t.StepDetails =
    type === StepTypes.MESSAGE_CREATION
      ? {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'msg-1' },
      }
      : { type: StepTypes.TOOL_CALLS, tool_calls: opts?.tool_calls ?? [] };
  return {
    type,
    id: opts?.id ?? 'run-step-1',
    index: opts?.index ?? 0,
    stepDetails,
    usage: null,
  };
}

function makeToolCall(id: string, name = 'calculator'): ToolCall {
  return { id, name, args: {}, type: 'tool_call' };
}

function makeToolCallChunk(opts?: {
  id?: string;
  name?: string;
  index?: number;
}): ToolCallChunk {
  return {
    id: opts?.id,
    name: opts?.name,
    args: '',
    index: opts?.index ?? 0,
    type: 'tool_call_chunk',
  };
}

const defaultMetadata = { run_id: 'test-run' };

describe('handleToolCallChunks', () => {
  let graph: MockGraph;

  beforeEach(() => {
    graph = createMockGraph();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('stamps every streamed fragment at receipt, before awaiting slow step creation', async () => {
    const step = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(step);
    jest.spyOn(Date, 'now').mockImplementation(() => 1_000);
    graph.dispatchRunStep.mockImplementation(async () => {
      jest.spyOn(Date, 'now').mockImplementation(() => 1_800);
      return 'step_with_tool';
    });
    const chunk = makeToolCallChunk({ index: 2 });
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: [chunk],
      metadata: defaultMetadata,
    });
    expect(graph.dispatchRunStepDelta).toHaveBeenCalledWith(
      'step_with_tool',
      { type: StepTypes.TOOL_CALLS, tool_calls: [chunk] },
      defaultMetadata,
      1_000
    );
  });

  it('creates TOOL_CALLS step when previous step is MESSAGE_CREATION', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    const chunks = [makeToolCallChunk({ index: 2 })];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    const dispatchCalls = graph.dispatchRunStep.mock.calls;
    expect(dispatchCalls).toHaveLength(1);
    expect(dispatchCalls[0][1]).toEqual(
      expect.objectContaining({ type: StepTypes.TOOL_CALLS })
    );
    expect(graph.messageStepHasToolCalls.has('prev-step-id')).toBe(true);
    expect(graph.dispatchRunStepDelta).toHaveBeenCalledTimes(1);
  });

  it('reuses existing TOOL_CALLS step without dispatching a new one', async () => {
    const toolStep = makeRunStep(StepTypes.TOOL_CALLS);
    graph.getRunStep.mockReturnValue(toolStep);

    const chunks = [makeToolCallChunk({ index: 2 })];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
    expect(graph.dispatchRunStepDelta).toHaveBeenCalledTimes(1);
    expect(graph.dispatchRunStepDelta).toHaveBeenCalledWith(
      'prev-step-id',
      expect.objectContaining({ type: StepTypes.TOOL_CALLS }),
      defaultMetadata,
      expect.any(Number)
    );
  });

  it('creates MESSAGE_CREATION when no previous step exists', async () => {
    let callCount = 0;
    graph.getStepIdByKey.mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        throw new Error('No step found');
      }
      return 'new-step-1';
    });
    graph.getRunStep.mockReturnValue(makeRunStep(StepTypes.MESSAGE_CREATION));

    const chunks = [makeToolCallChunk({ index: 0 })];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    const dispatchCalls = graph.dispatchRunStep.mock.calls;
    expect(dispatchCalls.length).toBeGreaterThanOrEqual(2);
    expect(dispatchCalls[0][1]).toEqual(
      expect.objectContaining({ type: StepTypes.MESSAGE_CREATION })
    );
    expect(dispatchCalls[1][1]).toEqual(
      expect.objectContaining({ type: StepTypes.TOOL_CALLS })
    );
  });

  it('skips TOOL_CALLS dispatch when already dispatched for this MESSAGE_CREATION', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);
    graph.messageStepHasToolCalls.set('prev-step-id', true);

    const chunks = [makeToolCallChunk({ index: 2 })];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
    expect(graph.dispatchRunStepDelta).toHaveBeenCalledTimes(1);
  });

  it('sanitizes empty string id and name to undefined', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    const chunk = makeToolCallChunk({ id: '', name: '' });
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: [chunk],
      metadata: defaultMetadata,
    });

    expect(chunk.id).toBeUndefined();
    expect(chunk.name).toBeUndefined();
  });

  it('populates tool_calls when chunk has valid id and name', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    const chunks = [
      makeToolCallChunk({ id: 'tooluse_abc', name: 'calculator', index: 2 }),
    ];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    const toolCallsArg = graph.dispatchRunStep.mock
      .calls[0][1] as t.ToolCallsDetails;
    expect(toolCallsArg.tool_calls).toEqual([
      expect.objectContaining({
        id: 'tooluse_abc',
        name: 'calculator',
        type: ToolCallTypes.TOOL_CALL,
      }),
    ]);
  });

  it('never dispatches empty text block alongside TOOL_CALLS step', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    const chunks = [
      makeToolCallChunk({ id: 'tooluse_abc', name: 'calculator', index: 2 }),
    ];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    const allDispatches = graph.dispatchRunStep.mock.calls;
    expect(allDispatches).toHaveLength(1);
    const stepDetails = allDispatches[0][1] as t.StepDetails;
    expect(stepDetails.type).toBe(StepTypes.TOOL_CALLS);
    expect(stepDetails).not.toHaveProperty('content');
    expect(stepDetails).not.toHaveProperty('text');
  });

  it('dispatches delta even when chunks lack id/name (Bedrock pattern)', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    const chunks = [makeToolCallChunk({ index: 2 })];
    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey: 'step-key',
      toolCallChunks: chunks,
      metadata: defaultMetadata,
    });

    expect(graph.dispatchRunStepDelta).toHaveBeenCalledTimes(1);
    const toolCallsArg = graph.dispatchRunStep.mock
      .calls[0][1] as t.ToolCallsDetails;
    expect(toolCallsArg.tool_calls).toEqual([]);
  });
});

describe('handleToolCalls', () => {
  let graph: MockGraph;

  beforeEach(() => {
    graph = createMockGraph();
  });

  it('returns early when metadata is missing', async () => {
    await handleToolCalls(
      [makeToolCall('id-1')],
      undefined,
      graph as unknown as StandardGraph
    );
    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
  });

  it('returns early when toolCalls is undefined', async () => {
    await handleToolCalls(
      undefined,
      defaultMetadata,
      graph as unknown as StandardGraph
    );
    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
  });

  it('returns early when toolCalls is empty', async () => {
    await handleToolCalls(
      [],
      defaultMetadata,
      graph as unknown as StandardGraph
    );
    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
  });

  it('skips tool call when id already in toolCallStepIds', async () => {
    graph.toolCallStepIds.set('id-1', 'existing-step');
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    await handleToolCalls(
      [makeToolCall('id-1')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
  });

  it('assigns fallback id when tool_call.id is undefined', async () => {
    const tc: ToolCall = {
      id: undefined as unknown as string,
      name: 'calc',
      args: {},
      type: 'tool_call',
    };
    graph.getStepIdByKey.mockImplementation(() => {
      throw new Error('no step');
    });

    await handleToolCalls(
      [tc],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(tc.id).toBeDefined();
    expect(tc.id!.startsWith('toolu_')).toBe(true);
    expect(graph.dispatchRunStep).toHaveBeenCalled();
  });

  it('flags messageStepHasToolCalls and dispatches TOOL_CALLS when prev step is MESSAGE_CREATION', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    await handleToolCalls(
      [makeToolCall('id-1')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(graph.messageStepHasToolCalls.get('prev-step-id')).toBe(true);
    const calls = graph.dispatchRunStep.mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0][1]).toEqual(
      expect.objectContaining({ type: StepTypes.TOOL_CALLS })
    );
  });

  it('creates MESSAGE_CREATION when no previous step exists', async () => {
    graph.getStepIdByKey.mockImplementation(() => {
      throw new Error('no step');
    });

    await handleToolCalls(
      [makeToolCall('id-1')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    const calls = graph.dispatchRunStep.mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[0][1]).toEqual(
      expect.objectContaining({ type: StepTypes.MESSAGE_CREATION })
    );
    expect(calls[1][1]).toEqual(
      expect.objectContaining({ type: StepTypes.TOOL_CALLS })
    );
  });

  it('reuses empty TOOL_CALLS step exactly once', async () => {
    const emptyToolStep = makeRunStep(StepTypes.TOOL_CALLS, {
      id: 'empty-step',
      tool_calls: [],
    });
    graph.getRunStep.mockReturnValue(emptyToolStep);
    graph.getStepIdByKey.mockReturnValue('empty-step-id');

    await handleToolCalls(
      [makeToolCall('id-1'), makeToolCall('id-2')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(graph.toolCallStepIds.get('id-1')).toBe('empty-step-id');

    const calls = graph.dispatchRunStep.mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0][1]).toEqual(
      expect.objectContaining({
        type: StepTypes.TOOL_CALLS,
        tool_calls: [expect.objectContaining({ id: 'id-2' })],
      })
    );
  });

  it('gives each parallel tool call its own step (3 tool calls)', async () => {
    const emptyToolStep = makeRunStep(StepTypes.TOOL_CALLS, { tool_calls: [] });
    graph.getStepIdByKey.mockReturnValue('chunk-step-id');

    let callCount = 0;
    graph.getRunStep.mockImplementation(() => {
      if (callCount === 0) {
        callCount++;
        return emptyToolStep;
      }
      return makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'prev',
            name: 'calc',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      });
    });

    await handleToolCalls(
      [makeToolCall('id-1'), makeToolCall('id-2'), makeToolCall('id-3')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(graph.toolCallStepIds.get('id-1')).toBe('chunk-step-id');

    const calls = graph.dispatchRunStep.mock.calls;
    expect(calls).toHaveLength(2);
    expect((calls[0][1] as t.ToolCallsDetails).tool_calls![0].id).toBe('id-2');
    expect((calls[1][1] as t.ToolCallsDetails).tool_calls![0].id).toBe('id-3');
  });

  it('never creates MESSAGE_CREATION for parallel tool calls after TOOL_CALLS prev', async () => {
    const emptyToolStep = makeRunStep(StepTypes.TOOL_CALLS, { tool_calls: [] });
    graph.getStepIdByKey.mockReturnValue('chunk-step-id');

    let callCount = 0;
    graph.getRunStep.mockImplementation(() => {
      if (callCount === 0) {
        callCount++;
        return emptyToolStep;
      }
      return makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'prev',
            name: 'calc',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      });
    });

    await handleToolCalls(
      [makeToolCall('id-1'), makeToolCall('id-2'), makeToolCall('id-3')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    const msgCreationCalls = graph.dispatchRunStep.mock.calls.filter(
      (call) => (call[1] as t.StepDetails).type === StepTypes.MESSAGE_CREATION
    );
    expect(msgCreationCalls).toHaveLength(0);
  });

  it('dispatches new TOOL_CALLS directly when prev TOOL_CALLS has existing data', async () => {
    const populatedToolStep = makeRunStep(StepTypes.TOOL_CALLS, {
      tool_calls: [
        {
          id: 'existing',
          name: 'calc',
          args: {},
          type: 'tool_call',
        } as t.AgentToolCall,
      ],
    });
    graph.getRunStep.mockReturnValue(populatedToolStep);

    await handleToolCalls(
      [makeToolCall('id-1')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    const calls = graph.dispatchRunStep.mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0][1]).toEqual(
      expect.objectContaining({ type: StepTypes.TOOL_CALLS })
    );
  });

  it('never dispatches empty text block with tool_call_ids (MESSAGE_CREATION path)', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    await handleToolCalls(
      [makeToolCall('id-1')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    for (const call of graph.dispatchRunStep.mock.calls) {
      const stepDetails = call[1] as t.StepDetails;
      if (stepDetails.type === StepTypes.TOOL_CALLS) {
        expect(stepDetails).not.toHaveProperty('content');
        expect(stepDetails).not.toHaveProperty('text');
      }
    }
  });

  it('never dispatches empty text block with tool_call_ids (no prev step path)', async () => {
    graph.getStepIdByKey.mockImplementation(() => {
      throw new Error('no step');
    });

    await handleToolCalls(
      [makeToolCall('id-1')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    for (const call of graph.dispatchRunStep.mock.calls) {
      const stepDetails = call[1] as t.StepDetails;
      if (stepDetails.type === StepTypes.TOOL_CALLS) {
        expect(stepDetails).not.toHaveProperty('content');
        expect(stepDetails).not.toHaveProperty('text');
      }
      if (stepDetails.type === StepTypes.MESSAGE_CREATION) {
        const msgDetails = stepDetails as t.MessageCreationDetails;
        expect(msgDetails.message_creation.message_id).toBeDefined();
        expect(msgDetails).not.toHaveProperty('tool_call_ids');
      }
    }
  });
});

describe('handleToolCallChunks + handleToolCalls integration', () => {
  let graph: MockGraph;
  const stepKey = 'step-key';

  beforeEach(() => {
    graph = createMockGraph();
  });

  it('Bedrock single tool: chunks create empty TOOL_CALLS, then handleToolCalls reuses it', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey,
      toolCallChunks: [makeToolCallChunk({ index: 2 })],
      metadata: defaultMetadata,
    });

    const chunkStepId = graph.dispatchRunStep.mock.results[0].value as string;
    const resolvedChunkStepId = await chunkStepId;

    const emptyToolStep = makeRunStep(StepTypes.TOOL_CALLS, { tool_calls: [] });
    graph.getRunStep.mockReturnValue(emptyToolStep);
    graph.getStepIdByKey.mockReturnValue(resolvedChunkStepId);
    graph.dispatchRunStep.mockClear();

    await handleToolCalls(
      [makeToolCall('tooluse_abc')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(graph.toolCallStepIds.get('tooluse_abc')).toBe(resolvedChunkStepId);
    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
  });

  it('Bedrock parallel: 3 chunks then 3 tool calls yields 3 unique step IDs', async () => {
    const msgStep = makeRunStep(StepTypes.MESSAGE_CREATION);
    graph.getRunStep.mockReturnValue(msgStep);

    await handleToolCallChunks({
      graph: graph as unknown as StandardGraph,
      stepKey,
      toolCallChunks: [makeToolCallChunk({ index: 2 })],
      metadata: defaultMetadata,
    });

    const chunkStepId = await (graph.dispatchRunStep.mock.results[0]
      .value as Promise<string>);

    const emptyToolStep = makeRunStep(StepTypes.TOOL_CALLS, { tool_calls: [] });
    graph.getStepIdByKey.mockReturnValue(chunkStepId);

    let callIdx = 0;
    graph.getRunStep.mockImplementation(() => {
      if (callIdx === 0) {
        callIdx++;
        return emptyToolStep;
      }
      return makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'prev',
            name: 'calc',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      });
    });
    graph.dispatchRunStep.mockClear();

    let newStepCounter = 10;
    graph.dispatchRunStep.mockImplementation(
      async () => `new-step-${++newStepCounter}`
    );

    await handleToolCalls(
      [makeToolCall('id-1'), makeToolCall('id-2'), makeToolCall('id-3')],
      defaultMetadata,
      graph as unknown as StandardGraph
    );

    expect(graph.toolCallStepIds.get('id-1')).toBe(chunkStepId);

    const dispatchedIds = graph.dispatchRunStep.mock.calls.map(
      (_, i) => graph.dispatchRunStep.mock.results[i].value
    );
    expect(dispatchedIds).toHaveLength(2);

    const allStepIds = new Set([
      chunkStepId,
      graph.toolCallStepIds.get('id-1'),
      ...graph.dispatchRunStep.mock.calls.map((call) => {
        const tc = (call[1] as t.ToolCallsDetails).tool_calls;
        return tc?.[0]?.id;
      }),
    ]);

    expect(graph.toolCallStepIds.get('id-1')).toBe(chunkStepId);
    expect(allStepIds.size).toBeGreaterThanOrEqual(2);

    const msgCreationCalls = graph.dispatchRunStep.mock.calls.filter(
      (call) => (call[1] as t.StepDetails).type === StepTypes.MESSAGE_CREATION
    );
    expect(msgCreationCalls).toHaveLength(0);
  });
});

describe('handleServerToolResult', () => {
  let graph: MockGraph;
  const anthropicContext = { provider: Providers.ANTHROPIC } as AgentContext;

  beforeEach(() => {
    graph = createMockGraph();
  });

  it('returns false when provider is not Anthropic', async () => {
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [{ type: 'tool_result', tool_use_id: 'tu-1', content: 'ok' }],
      agentContext: { provider: Providers.OPENAI } as AgentContext,
    });
    expect(result).toBe(false);
  });

  it('returns false when content is a string', async () => {
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: 'plain text',
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
  });

  it('returns false when content is null/undefined', async () => {
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: undefined,
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
  });

  it('returns false when content is empty array', async () => {
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
  });

  it('returns false when single content item has no tool_use_id', async () => {
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [{ type: 'tool_result', content: 'ok' } as t.ToolResultContent],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
  });

  it('skips content parts with empty tool_use_id', async () => {
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [
        { type: 'tool_result', tool_use_id: '', content: 'ok' },
        { type: 'tool_result', tool_use_id: 'tu-valid', content: 'ok' },
      ] as t.MessageContentComplex[],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
  });

  it('warns and skips when toolCallStepIds has no mapping for tool_use_id', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [
        { type: 'tool_result', tool_use_id: 'tu-missing', content: 'ok' },
      ] as t.MessageContentComplex[],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('tu-missing'));
    warnSpy.mockRestore();
  });

  it('warns when run step does not exist for stepId', async () => {
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(undefined);
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [
        { type: 'tool_result', tool_use_id: 'tu-1', content: 'ok' },
      ] as t.MessageContentComplex[],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('does not exist')
    );
    warnSpy.mockRestore();
  });

  it('warns when run step is not a TOOL_CALLS type', async () => {
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(makeRunStep(StepTypes.MESSAGE_CREATION));
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [
        { type: 'tool_result', tool_use_id: 'tu-1', content: 'ok' },
      ] as t.MessageContentComplex[],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('not a tool call step')
    );
    warnSpy.mockRestore();
  });

  it('skips when no matching tool call found in step details', async () => {
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(
      makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'tu-other',
            name: 'calc',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      })
    );

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [
        { type: 'tool_result', tool_use_id: 'tu-1', content: 'ok' },
      ] as t.MessageContentComplex[],
      agentContext: anthropicContext,
    });
    expect(result).toBe(false);
  });

  it('returns true and sets skipHandling when a valid tool result is found', async () => {
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(
      makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'tu-1',
            name: 'calc',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      })
    );

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [
        { type: 'tool_result', tool_use_id: 'tu-1', content: 'ok' },
      ] as t.MessageContentComplex[],
      agentContext: anthropicContext,
    });
    expect(result).toBe(true);
  });

  it('calls handleAnthropicSearchResults for web_search_result type', async () => {
    const mockToolEndHandle = jest
      .fn<(...args: unknown[]) => Promise<void>>()
      .mockResolvedValue(undefined);
    graph.handlerRegistry = {
      getHandler: jest.fn().mockReturnValue({ handle: mockToolEndHandle }),
    };
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(
      makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'tu-1',
            name: 'web_search',
            args: { query: 'test' },
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      })
    );

    const webSearchContent: t.ToolResultContent = {
      type: 'web_search_result',
      tool_use_id: 'tu-1',
      content: [
        {
          type: 'web_search_result',
          url: 'https://example.com',
          title: 'Example',
          encrypted_index: 'abc',
          page_age: '2024-01-01',
        },
      ],
    };

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [webSearchContent] as t.MessageContentComplex[],
      metadata: defaultMetadata,
      agentContext: anthropicContext,
    });

    expect(result).toBe(true);
    expect(mockToolEndHandle).toHaveBeenCalledWith(
      GraphEvents.TOOL_END,
      expect.objectContaining({ input: { query: 'test' } }),
      defaultMetadata,
      graph
    );
    expect(graph.invokedToolIds).toBeDefined();
    expect(graph.invokedToolIds!.has('tu-1')).toBe(true);
  });

  it('initializes invokedToolIds set when null', async () => {
    const mockToolEndHandle = jest
      .fn<(...args: unknown[]) => Promise<void>>()
      .mockResolvedValue(undefined);
    graph.handlerRegistry = {
      getHandler: jest.fn().mockReturnValue({ handle: mockToolEndHandle }),
    };
    graph.invokedToolIds = undefined;
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(
      makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'tu-1',
            name: 'web_search',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      })
    );

    const webSearchContent: t.ToolResultContent = {
      type: 'web_search_tool_result',
      tool_use_id: 'tu-1',
      content: [
        {
          type: 'web_search_result',
          url: 'https://example.com',
          title: 'Test',
          encrypted_index: 'x',
        },
      ],
    };

    await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [webSearchContent] as t.MessageContentComplex[],
      metadata: defaultMetadata,
      agentContext: anthropicContext,
    });

    expect(graph.invokedToolIds).toBeInstanceOf(Set);
    expect(graph.invokedToolIds!.has('tu-1')).toBe(true);
  });

  it('warns when web search content is not an array', async () => {
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(
      makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'tu-1',
            name: 'web_search',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      })
    );
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const webSearchContent: t.ToolResultContent = {
      type: 'web_search_result',
      tool_use_id: 'tu-1',
      content: 'not an array',
    };

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [webSearchContent] as t.MessageContentComplex[],
      metadata: defaultMetadata,
      agentContext: anthropicContext,
    });

    expect(result).toBe(true);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('Expected content to be an array')
    );
    warnSpy.mockRestore();
  });

  it('warns when content is not an Anthropic web search result', async () => {
    graph.toolCallStepIds.set('tu-1', 'step-1');
    graph.getRunStep.mockReturnValue(
      makeRunStep(StepTypes.TOOL_CALLS, {
        tool_calls: [
          {
            id: 'tu-1',
            name: 'web_search',
            args: {},
            type: 'tool_call',
          } as t.AgentToolCall,
        ],
      })
    );
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const webSearchContent: t.ToolResultContent = {
      type: 'web_search_result',
      tool_use_id: 'tu-1',
      content: [{ type: 'text', text: 'not a search result' }],
    };

    const result = await handleServerToolResult({
      graph: graph as unknown as StandardGraph,
      content: [webSearchContent] as t.MessageContentComplex[],
      metadata: defaultMetadata,
      agentContext: anthropicContext,
    });

    expect(result).toBe(true);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'Expected content to be an Anthropic web search result'
      )
    );
    warnSpy.mockRestore();
  });
});

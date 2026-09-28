/* eslint-disable no-console */
// src/tools/handlers.ts
import { nanoid } from 'nanoid';
import { ToolMessage } from '@langchain/core/messages';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { AnthropicWebSearchResultBlockParam } from '@/llm/anthropic/types';
import type { Graph, MultiAgentGraph, StandardGraph } from '@/graphs';
import type { AgentContext } from '@/agents/AgentContext';
import type * as t from '@/types';
import {
  coerceAnthropicSearchResults,
  isAnthropicWebSearchResult,
} from '@/tools/search/anthropic';
import {
  ToolCallTypes,
  GraphEvents,
  StepTypes,
  Providers,
  Constants,
} from '@/common';
import { formatResultsForLLM } from '@/tools/search/format';
import { getMessageId } from '@/messages';

export async function handleToolCallChunks({
  graph,
  stepKey,
  toolCallChunks,
  metadata,
  observedAt = Date.now(),
}: {
  graph: StandardGraph | MultiAgentGraph;
  stepKey: string;
  toolCallChunks: ToolCallChunk[];
  metadata?: Record<string, unknown>;
  observedAt?: number;
}): Promise<void> {
  let prevStepId: string;
  let prevRunStep: t.RunStep | undefined;
  try {
    prevStepId = graph.getStepIdByKey(stepKey);
    prevRunStep = graph.getRunStep(prevStepId);
  } catch {
    /** Edge Case: If no previous step exists, create a new message creation step */
    const message_id = getMessageId(stepKey, graph, true) ?? '';
    prevStepId = await graph.dispatchRunStep(
      stepKey,
      {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: {
          message_id,
        },
      },
      metadata
    );
    prevRunStep = graph.getRunStep(prevStepId);
  }

  const _stepId = graph.getStepIdByKey(stepKey);

  /** Edge Case: Tool Call Run Step or `tool_call_ids` never dispatched */
  const tool_calls: ToolCall[] | undefined =
    prevStepId && prevRunStep && prevRunStep.type === StepTypes.MESSAGE_CREATION
      ? []
      : undefined;

  /** Edge Case: `id` and `name` fields cannot be empty strings */
  for (const toolCallChunk of toolCallChunks) {
    if (toolCallChunk.name === '') {
      toolCallChunk.name = undefined;
    }
    if (toolCallChunk.id === '') {
      toolCallChunk.id = undefined;
    } else if (
      tool_calls != null &&
      toolCallChunk.id != null &&
      toolCallChunk.name != null
    ) {
      tool_calls.push({
        args: {},
        id: toolCallChunk.id,
        name: toolCallChunk.name,
        type: ToolCallTypes.TOOL_CALL,
      });
    }
  }

  let stepId: string = _stepId;
  const alreadyDispatched =
    prevRunStep?.type === StepTypes.MESSAGE_CREATION &&
    graph.messageStepHasToolCalls.has(prevStepId);

  if (prevRunStep?.type === StepTypes.TOOL_CALLS) {
    /**
     * If previous step is already a tool_calls step, use that step ID
     * This ensures tool call deltas are dispatched to the correct step
     */
    stepId = prevStepId;
  } else if (
    !alreadyDispatched &&
    prevRunStep?.type === StepTypes.MESSAGE_CREATION
  ) {
    /**
     * Create tool_calls step as soon as we receive the first tool call chunk
     * This ensures deltas are always associated with the correct step
     *
     * NOTE: We do NOT dispatch an empty text block here because:
     * - Empty text blocks cause providers (Anthropic, Bedrock) to reject messages
     * - The tool_calls themselves are sufficient for the step
     * - Empty content with tool_call_ids gets stored in conversation history
     *   and causes "messages must have non-empty content" errors on replay
     */
    graph.messageStepHasToolCalls.set(prevStepId, true);
    stepId = await graph.dispatchRunStep(
      stepKey,
      {
        type: StepTypes.TOOL_CALLS,
        tool_calls: tool_calls ?? [],
      },
      metadata
    );
  }

  await graph.dispatchRunStepDelta(
    stepId,
    {
      type: StepTypes.TOOL_CALLS,
      tool_calls: toolCallChunks,
    },
    metadata,
    observedAt
  );
}

export const handleToolCalls = async (
  toolCalls?: ToolCall[],
  metadata?: Record<string, unknown>,
  graph?: Graph | StandardGraph | MultiAgentGraph
): Promise<void> => {
  if (!graph || !metadata) {
    console.warn('Graph or metadata not found in `handleToolCalls`');
    return;
  }

  if (!toolCalls) {
    return;
  }

  if (toolCalls.length === 0) {
    return;
  }

  const stepKey = graph.getStepKey(metadata);

  /**
   * Track whether we've already reused an empty TOOL_CALLS step created by
   * handleToolCallChunks during streaming. Only reuse it once (for the first
   * tool call); subsequent parallel tool calls must create their own steps.
   */
  let reusedChunkStepId: string | undefined;

  for (const tool_call of toolCalls) {
    const toolCallId = tool_call.id ?? `toolu_${nanoid()}`;
    tool_call.id = toolCallId;
    if (!toolCallId || graph.toolCallStepIds.has(toolCallId)) {
      continue;
    }

    let prevStepId = '';
    let prevRunStep: t.RunStep | undefined;
    try {
      prevStepId = graph.getStepIdByKey(stepKey);
      prevRunStep = graph.getRunStep(prevStepId);
    } catch {
      // no previous step
    }

    /**
     * If the previous step is TOOL_CALLS (from handleToolCallChunks or a prior
     * iteration), either reuse it (if empty) or dispatch a new TOOL_CALLS step
     * directly — skip the intermediate MESSAGE_CREATION to avoid orphaned gaps.
     */
    if (prevRunStep?.type === StepTypes.TOOL_CALLS) {
      const details = prevRunStep.stepDetails as t.ToolCallsDetails;
      const isEmpty = !details.tool_calls || details.tool_calls.length === 0;
      if (isEmpty && prevStepId !== reusedChunkStepId) {
        graph.toolCallStepIds.set(toolCallId, prevStepId);
        graph.registerPendingToolCall(toolCallId, prevStepId);
        reusedChunkStepId = prevStepId;
        continue;
      }
      await graph.dispatchRunStep(
        stepKey,
        { type: StepTypes.TOOL_CALLS, tool_calls: [tool_call] },
        metadata
      );
      continue;
    }

    /**
     * NOTE: We do NOT dispatch empty text blocks with tool_call_ids because:
     * - Empty text blocks cause providers (Anthropic, Bedrock) to reject messages
     * - They get stored in conversation history and cause errors on replay:
     *   "messages must have non-empty content" (Anthropic)
     *   "The content field in the Message object is empty" (Bedrock)
     * - The tool_calls themselves are sufficient
     */
    if (prevStepId && prevRunStep) {
      graph.messageStepHasToolCalls.set(prevStepId, true);
    } else if (!prevRunStep) {
      const messageId = getMessageId(stepKey, graph, true) ?? '';
      const stepId = await graph.dispatchRunStep(
        stepKey,
        {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: {
            message_id: messageId,
          },
        },
        metadata
      );
      graph.messageStepHasToolCalls.set(stepId, true);
    }

    await graph.dispatchRunStep(
      stepKey,
      {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [tool_call],
      },
      metadata
    );
  }
};

export const toolResultTypes = new Set([
  // 'tool_use',
  // 'server_tool_use',
  // 'input_json_delta',
  'tool_result',
  'web_search_result',
  'web_search_tool_result',
]);

/**
 * Handles the result of a server tool call; in other words, a provider's built-in tool.
 * As of 2025-07-06, only Anthropic handles server tool calls with this pattern.
 */
export async function handleServerToolResult({
  graph,
  content,
  metadata,
  agentContext,
}: {
  graph: StandardGraph | MultiAgentGraph;
  content?: string | t.MessageContentComplex[];
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
}): Promise<boolean> {
  let skipHandling = false;
  if (agentContext?.provider !== Providers.ANTHROPIC) {
    return skipHandling;
  }
  if (
    typeof content === 'string' ||
    content == null ||
    content.length === 0 ||
    (content.length === 1 &&
      (content[0] as t.ToolResultContent).tool_use_id == null)
  ) {
    return skipHandling;
  }

  for (const contentPart of content) {
    const toolUseId = (contentPart as t.ToolResultContent).tool_use_id;
    if (toolUseId == null || toolUseId === '') {
      continue;
    }
    const stepId = graph.toolCallStepIds.get(toolUseId);
    if (stepId == null || stepId === '') {
      console.warn(
        `Tool use ID ${toolUseId} not found in graph, cannot dispatch tool result.`
      );
      continue;
    }
    const runStep = graph.getRunStep(stepId);
    if (!runStep) {
      console.warn(
        `Run step for ${stepId} does not exist, cannot dispatch tool result.`
      );
      continue;
    } else if (runStep.type !== StepTypes.TOOL_CALLS) {
      console.warn(
        `Run step for ${stepId} is not a tool call step, cannot dispatch tool result.`
      );
      continue;
    }

    const toolCall =
      runStep.stepDetails.type === StepTypes.TOOL_CALLS
        ? (runStep.stepDetails.tool_calls?.find(
          (toolCall) => toolCall.id === toolUseId
        ) as ToolCall)
        : undefined;

    if (!toolCall) {
      continue;
    }

    if (
      contentPart.type === 'web_search_result' ||
      contentPart.type === 'web_search_tool_result'
    ) {
      await handleAnthropicSearchResults({
        contentPart: contentPart as t.ToolResultContent,
        toolCall,
        metadata,
        graph,
      });
    }

    if (!skipHandling) {
      skipHandling = true;
    }
  }

  return skipHandling;
}

async function handleAnthropicSearchResults({
  contentPart,
  toolCall,
  metadata,
  graph,
}: {
  contentPart: t.ToolResultContent;
  toolCall: Partial<ToolCall>;
  metadata?: Record<string, unknown>;
  graph: StandardGraph | MultiAgentGraph;
}): Promise<void> {
  if (!Array.isArray(contentPart.content)) {
    console.warn(
      `Expected content to be an array, got ${typeof contentPart.content}`
    );
    return;
  }

  if (!isAnthropicWebSearchResult(contentPart.content[0])) {
    console.warn(
      `Expected content to be an Anthropic web search result, got ${JSON.stringify(
        contentPart.content
      )}`
    );
    return;
  }

  const turn = graph.invokedToolIds?.size ?? 0;
  const searchResultData = coerceAnthropicSearchResults({
    turn,
    results: contentPart.content as AnthropicWebSearchResultBlockParam[],
  });

  const name = toolCall.name;
  const input = toolCall.args ?? {};
  const artifact = {
    [Constants.WEB_SEARCH]: searchResultData,
  };
  const { output: formattedOutput } = formatResultsForLLM(
    turn,
    searchResultData
  );
  const output = new ToolMessage({
    name,
    artifact,
    content: formattedOutput,
    tool_call_id: toolCall.id!,
  });
  const toolEndData: t.ToolEndData = {
    input,
    output,
  };
  await graph.handlerRegistry
    ?.getHandler(GraphEvents.TOOL_END)
    ?.handle(GraphEvents.TOOL_END, toolEndData, metadata, graph);

  if (graph.invokedToolIds == null) {
    graph.invokedToolIds = new Set<string>();
  }

  graph.invokedToolIds.add(toolCall.id!);
}

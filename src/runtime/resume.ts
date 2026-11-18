import type { EventHandler, MessageContentComplex, RunStep } from '@/index';
import type { RemoteSdkSnapshot } from './types';
import { ContentTypes, GraphEvents, convertMessagesToContent } from '@/index';
import { AIMessage, isAIMessage, isToolMessage } from '@/langchain/messages';

type ResumableRunStep = {
  id: string;
  index: number;
  stepDetails: {
    type: string;
    tool_calls?: readonly { id?: string }[];
  };
};

export function normalizeResumeRunStepIndices<T extends ResumableRunStep>(
  runSteps: readonly T[],
  seedContent: readonly (
    | { type?: string; tool_call?: { id?: string } }
    | undefined
  )[] = []
): T[] {
  const toolCallIndices = new Map<string, number>();
  seedContent.forEach((part, index) => {
    const toolCallId = part?.tool_call?.id;
    if (part?.type === 'tool_call' && typeof toolCallId === 'string') {
      toolCallIndices.set(toolCallId, index);
    }
  });

  return runSteps.map((runStep) => {
    if (runStep.stepDetails.type !== 'tool_calls') {
      return runStep;
    }
    const contentIndex = runStep.stepDetails.tool_calls
      ?.map((toolCall) =>
        toolCall.id != null && toolCall.id.length > 0
          ? toolCallIndices.get(toolCall.id)
          : undefined
      )
      .find((index) => index != null);
    return contentIndex != null && contentIndex !== runStep.index
      ? { ...runStep, index: contentIndex }
      : runStep;
  });
}

/**
 * Restore the streamed run-step sidecars that a fresh SDK Run cannot recover
 * from the LangGraph checkpoint by itself.
 *
 * Human-review resume can happen in a later request or process. The checkpoint
 * restarts directly inside ToolNode, so it does not replay ON_RUN_STEP before
 * dispatching ON_RUN_STEP_COMPLETED. Seeding both maps lets the ToolNode emit
 * the original step id and lets the content aggregator resolve that id back to
 * the already-rendered tool card.
 */
export function hydrateResumeRunSteps(
  runSteps: readonly RunStep[],
  stepMap: Map<string, RunStep | undefined> | undefined,
  graph: { toolCallStepIds?: Map<string, string> } | null | undefined,
  seedContent: readonly (
    | { type?: string; tool_call?: { id?: string } }
    | undefined
  )[] = []
): void {
  for (const runStep of normalizeResumeRunStepIndices(runSteps, seedContent)) {
    if (runStep.id.length === 0) {
      continue;
    }
    stepMap?.set(runStep.id, runStep);
    const stepDetails: ResumableRunStep['stepDetails'] = runStep.stepDetails;
    if (stepDetails.type !== 'tool_calls') {
      continue;
    }
    for (const toolCall of stepDetails.tool_calls ?? []) {
      if (toolCall.id != null && toolCall.id.length > 0) {
        graph?.toolCallStepIds?.set(toolCall.id, runStep.id);
      }
    }
  }
}

/**
 * Wrap a resume run's event handlers so every content index the rebuilt graph
 * emits is shifted past the pre-pause content.
 *
 * WHY: a resumed run rebuilds the graph from the checkpoint, and the fresh
 * graph assigns content indices from its own empty `contentData` — starting at
 * 0. The host, meanwhile, seeds the (also fresh) content aggregator with the
 * pre-pause parts, which occupy exactly those low indices. Without an offset
 * the resumed model turn collides with the seed: when the types match at an
 * index the new text silently MERGES into a pre-pause part, and when they
 * don't (e.g. a reasoning/`think` part at index 0 with Anthropic models) every
 * delta is dropped with `Content type mismatch` — the entire post-resume
 * output vanishes from both the live stream and the saved message.
 *
 * The index enters the pipeline at exactly one point: `ON_RUN_STEP`'s payload
 * (the `RunStep`, whose `index` every subsequent delta resolves through the
 * aggregator's `stepMap`). `ON_AGENT_UPDATE` carries its own inline index and
 * is offset likewise. All other handlers pass through untouched — same object
 * references, so stateful handler instances keep working.
 */
export function createContentIndexOffsetHandlers(
  handlers: Record<string, EventHandler> | undefined,
  seedContent: Array<{
    type?: string;
    tool_call?: { id?: string; output?: unknown };
  }> = []
): Record<string, EventHandler> | undefined {
  const offset = seedContent.length;
  if (handlers == null || !(offset > 0)) {
    return handlers;
  }

  /**
   * Resumed tool steps for calls the PAUSED turn already rendered must land
   * back on their seeded slot — not a fresh offset slot — or the original
   * part stays unresolved while a duplicate completed one appears after the
   * seed (and its output never attaches). Map unresolved seeded tool_calls by
   * id so the resume pass's re-execution (approval flows re-run the approved
   * tool; ask re-runs its body) rebinds to the right index.
   */
  const seededToolCallIndex = new Map<string, number>();
  seedContent.forEach((part, index) => {
    const toolCall = part.tool_call;
    /**
     * EVERY seeded id maps — including parts already carrying an output. Tool
     * call ids are minted per call by the provider, so a resumed step bearing
     * a seeded id can only be the interrupted batch re-executing (the resume
     * controller pre-stamps the ask part's answer onto the seed, which must
     * not exile its re-run step to a duplicate offset slot).
     */
    if (part.type === 'tool_call' && typeof toolCall?.id === 'string') {
      seededToolCallIndex.set(toolCall.id, index);
    }
  });

  const wrapped: Record<string, EventHandler> = { ...handlers };

  const runStepHandler = (handlers as Partial<typeof handlers>)[
    GraphEvents.ON_RUN_STEP
  ];
  if (runStepHandler) {
    wrapped[GraphEvents.ON_RUN_STEP] = {
      handle: (event, data, metadata, graph): void | Promise<void> => {
        const runStep = data as
          | {
              index?: number;
              stepDetails?: {
                type?: string;
                tool_calls?: Array<{ id?: string }>;
              };
            }
          | undefined;
        if (runStep == null || typeof runStep.index !== 'number') {
          return runStepHandler.handle(event, data, metadata, graph);
        }
        const seededIndex =
          runStep.stepDetails?.type === 'tool_calls'
            ? runStep.stepDetails.tool_calls
              ?.map((call) =>
                call.id != null && call.id.length > 0
                  ? seededToolCallIndex.get(call.id)
                  : undefined
              )
              .find((index) => index != null)
            : undefined;
        const shifted = {
          ...runStep,
          index: seededIndex ?? runStep.index + offset,
        };
        return runStepHandler.handle(
          event,
          shifted as typeof data,
          metadata,
          graph
        );
      },
    };
  }

  const agentUpdateHandler = (handlers as Partial<typeof handlers>)[
    GraphEvents.ON_AGENT_UPDATE
  ];
  if (agentUpdateHandler) {
    wrapped[GraphEvents.ON_AGENT_UPDATE] = {
      handle: (event, data, metadata, graph): void | Promise<void> => {
        const update = data as
          | { agent_update?: { index?: number } }
          | undefined;
        const shifted =
          update?.agent_update != null &&
          typeof update.agent_update.index === 'number'
            ? {
              ...update,
              agent_update: {
                ...update.agent_update,
                index: update.agent_update.index + offset,
              },
            }
            : data;
        return agentUpdateHandler.handle(
          event,
          shifted as typeof data,
          metadata,
          graph
        );
      },
    };
  }

  return wrapped;
}

/** Merge a worker continuation with the host’s already rendered prefix. */
export function reconcileRemoteResumeContent(
  seededProjection: readonly (MessageContentComplex | undefined)[],
  snapshot: RemoteSdkSnapshot
): MessageContentComplex[] {
  const content = seededProjection.filter((part) => part != null);
  const toolSlots = new Map<string, number>();
  for (const [index, part] of content.entries()) {
    if (part.type !== ContentTypes.TOOL_CALL) continue;
    const id = part.tool_call?.id;
    if (id != null && id !== '') toolSlots.set(id, index);
  }
  if (toolSlots.size === 0 || snapshot.messages == null)
    return content.concat(snapshot.content ?? []);

  let replayed = false;
  const messages: NonNullable<RemoteSdkSnapshot['messages']> = [];
  for (const message of snapshot.messages) {
    if (isAIMessage(message) && message.tool_calls != null) {
      const calls = message.tool_calls.filter(
        (call) => call.id == null || !toolSlots.has(call.id)
      );
      if (calls.length !== message.tool_calls.length) {
        replayed = true;
        // Retain declarations for calls in the replayed batch without a seeded card.
        if (calls.length > 0)
          messages.push(new AIMessage({ content: '', tool_calls: calls }));
        continue;
      }
    }
    if (!isToolMessage(message)) {
      messages.push(message);
      continue;
    }
    const index = toolSlots.get(message.tool_call_id);
    const part = index === undefined ? undefined : content[index];
    if (
      index === undefined ||
      part?.type !== ContentTypes.TOOL_CALL ||
      part.tool_call == null
    ) {
      messages.push(message);
      continue;
    }
    replayed = true;
    content[index] = {
      ...part,
      tool_call: { ...part.tool_call, output: message.content },
    };
  }
  return content.concat(
    replayed ? convertMessagesToContent(messages) : (snapshot.content ?? [])
  );
}

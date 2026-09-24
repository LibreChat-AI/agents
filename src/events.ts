/* eslint-disable no-console */
// src/events.ts
import type {
  BaseMessageFields,
  UsageMetadata,
} from '@langchain/core/messages';
import type { Logger } from 'winston';
import type { MultiAgentGraph, StandardGraph } from '@/graphs';
import type * as t from '@/types';
import { dispatchesChatModelStream, SDK_STREAM_DISPATCH } from '@/stream';
import { Constants, GraphEvents } from '@/common';

export class HandlerRegistry {
  private handlers: Map<string, t.EventHandler> = new Map();

  register(eventType: string, handler: t.EventHandler): void {
    this.handlers.set(eventType, handler);
  }

  getHandler(eventType: string): t.EventHandler | undefined {
    return this.handlers.get(eventType);
  }
}

export function composeEventHandlers(
  ...handlerSets: Array<Record<string, t.EventHandler> | undefined>
): Record<string, t.EventHandler> {
  const composed: Partial<Record<string, t.EventHandler>> = {};
  const acceptedObservers: t.EventHandler[] = [];

  for (const handlerSet of handlerSets) {
    if (!handlerSet) {
      continue;
    }
    for (const [eventType, handler] of Object.entries(handlerSet)) {
      if (eventType === GraphEvents.ON_MODEL_RESPONSE) {
        acceptedObservers.push(handler);
        composed[eventType] = handler;
        continue;
      }
      const previous = composed[eventType];
      if (previous === undefined) {
        composed[eventType] = handler;
        continue;
      }
      const wrapper: t.EventHandler = {
        handle: async (
          ...args: Parameters<t.EventHandler['handle']>
        ): Promise<void> => {
          await previous.handle(...args);
          await handler.handle(...args);
        },
      };
      /**
       * Carry the stream-dispatch brand across the wrapper. Without this a
       * composed handler fails every capability check made about it — notably
       * the one deciding whether a run may seal — even though it still drives
       * the SDK's content-part dispatch.
       */
      if (
        dispatchesChatModelStream(previous) ||
        dispatchesChatModelStream(handler)
      ) {
        Object.defineProperty(wrapper, SDK_STREAM_DISPATCH, { value: true });
      }
      composed[eventType] = wrapper;
    }
  }

  if (acceptedObservers.length > 1) {
    composed[GraphEvents.ON_MODEL_RESPONSE] = {
      handle: async (event, data, metadata, graph): Promise<void> => {
        // Clone from the graph-owned snapshot, not from any preceding observer's
        // possibly mutated argument tree. One bounded copy per observer.
        for (const observer of acceptedObservers) {
          await observer.handle(event, structuredClone(data), metadata, graph);
        }
      },
    };
  }

  return composed as Record<string, t.EventHandler>;
}

export class ModelEndHandler implements t.EventHandler {
  collectedUsage?: UsageMetadata[];
  constructor(collectedUsage?: UsageMetadata[]) {
    if (collectedUsage && !Array.isArray(collectedUsage)) {
      throw new Error('collectedUsage must be an array');
    }
    this.collectedUsage = collectedUsage;
  }

  async handle(
    event: string,
    data: t.ModelEndData,
    metadata?: Record<string, unknown>,
    graph?: StandardGraph | MultiAgentGraph
  ): Promise<void> {
    if (!graph || !metadata) {
      console.warn(`Graph or metadata not found in ${event} event`);
      return;
    }

    const usage = data?.output?.usage_metadata;
    if (usage != null && this.collectedUsage != null) {
      this.collectedUsage.push(usage);
    }
  }
}

export class ToolEndHandler implements t.EventHandler {
  private callback?: t.ToolEndCallback;
  private logger?: Logger;
  constructor(callback?: t.ToolEndCallback, logger?: Logger) {
    this.callback = callback;
    this.logger = logger;
  }

  /**
   * Handles on_tool_end events from the for-await stream consumer.
   *
   * This handler is now purely a consumer callback — tool completion
   * (ON_RUN_STEP_COMPLETED dispatch + session context storage) is handled
   * in graph context by ToolNode directly, eliminating the race between
   * the stream consumer and graph execution.
   */
  async handle(
    event: string,
    data: t.StreamEventData | undefined,
    metadata?: Record<string, unknown>,
    graph?: StandardGraph | MultiAgentGraph
  ): Promise<void> {
    try {
      if (!graph || !metadata) {
        if (this.logger) {
          this.logger.warn(`Graph or metadata not found in ${event} event`);
        } else {
          console.warn(`Graph or metadata not found in ${event} event`);
        }
        return;
      }

      const toolEndData = data as t.ToolEndData | undefined;
      if (!toolEndData?.output) {
        if (this.logger) {
          this.logger.warn('No output found in tool_end event');
        } else {
          console.warn('No output found in tool_end event');
        }
        return;
      }

      if (
        metadata[Constants.PROGRAMMATIC_TOOL_CALLING] === true ||
        metadata[Constants.BASH_PROGRAMMATIC_TOOL_CALLING] === true
      ) {
        return;
      }

      if (this.callback) {
        await this.callback(toolEndData, metadata);
      }
    } catch (error) {
      if (this.logger) {
        this.logger.error('Error handling tool_end event:', error);
      } else {
        console.error('Error handling tool_end event:', error);
      }
    }
  }
}

export class TestLLMStreamHandler implements t.EventHandler {
  handle(event: string, data: t.StreamEventData | undefined): void {
    const chunk = data?.chunk;
    const isMessageChunk = !!(chunk && 'message' in chunk);
    const msg = isMessageChunk ? chunk.message : undefined;
    if (msg && msg.tool_call_chunks && msg.tool_call_chunks.length > 0) {
      console.log(msg.tool_call_chunks);
    } else if (msg && typeof msg.content === 'string') {
      process.stdout.write(msg.content);
    }
  }
}

export class TestChatStreamHandler implements t.EventHandler {
  handle(event: string, data: t.StreamEventData | undefined): void {
    const chunk = data?.chunk;
    const isContentChunk = !!(chunk && 'content' in chunk);
    if (!isContentChunk) {
      return;
    }

    const content = chunk.content;

    if (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) {
      console.dir(chunk.tool_call_chunks, { depth: null });
    }

    if (typeof content === 'string') {
      process.stdout.write(content);
    } else {
      console.dir(content, { depth: null });
    }
  }
}

export class LLMStreamHandler implements t.EventHandler {
  handle(
    event: string,
    data: t.StreamEventData | undefined,
    metadata?: Record<string, unknown>
  ): void {
    const chunk = data?.chunk;
    const isMessageChunk = !!(chunk && 'message' in chunk);
    const msg = isMessageChunk ? chunk.message : undefined;
    if (metadata) {
      console.log(metadata);
    }
    if (msg && msg.tool_call_chunks && msg.tool_call_chunks.length > 0) {
      console.log(msg.tool_call_chunks);
    } else if (msg && typeof msg.content === 'string') {
      process.stdout.write(msg.content);
    }
  }
}

export const createMetadataAggregator = (
  _collected?: Record<
    string,
    NonNullable<BaseMessageFields['response_metadata']>
  >[]
): t.MetadataAggregatorResult => {
  const collected = _collected || [];

  const handleLLMEnd: t.HandleLLMEnd = (output) => {
    const { generations } = output;
    const lastMessageOutput = (
      generations[generations.length - 1] as
        | (t.StreamGeneration | undefined)[]
        | undefined
    )?.[0];
    if (!lastMessageOutput) {
      return;
    }
    const { message } = lastMessageOutput;
    if (message?.response_metadata) {
      collected.push(message.response_metadata);
    }
  };

  return { handleLLMEnd, collected };
};

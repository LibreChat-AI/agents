/**
 * Converts a raw Anthropic SSE event stream into LangChain ChatModelStreamEvents.
 *
 * @module
 */

import type {
  ChatModelStreamEvent,
  ContentBlockDelta,
  FinishReason,
} from '@langchain/core/language_models/event';
import type { ContentBlock, UsageMetadata } from '@langchain/core/messages';
import type {
  AnthropicCompactionBlock,
  AnthropicCompactionContentBlockDelta,
  AnthropicMessageStreamEvent,
} from '../types';
import type { AnthropicUsageData } from './message_outputs';
import { isAnthropicServerToolResultType } from '@/messages/anthropicServerTools';
import { getAnthropicUsageMetadata } from './message_outputs';

type AnthropicContentBlock =
  | Extract<
      AnthropicMessageStreamEvent,
      { type: 'content_block_start' }
    >['content_block']
  | AnthropicCompactionBlock;

type AnthropicContentDelta =
  | Extract<
      AnthropicMessageStreamEvent,
      { type: 'content_block_delta' }
    >['delta']
  | AnthropicCompactionContentBlockDelta;

interface AnthropicStreamErrorEvent {
  type: 'error';
  error: {
    type: string;
    message: string;
  };
  request_id?: string | null;
}

type AnthropicStreamInputEvent =
  | AnthropicMessageStreamEvent
  | AnthropicStreamErrorEvent;

type BlockAccumulator = Record<string, unknown>;

interface AnthropicEventUsage {
  inputTokens: number;
  cacheCreationInputTokens: number;
  cacheReadInputTokens: number;
  outputTokens: number;
}

// ─── Public API ─────────────────────────────────────────────────

export interface ConvertAnthropicStreamOptions {
  streamUsage?: boolean;
}

/**
 * Convert an async iterable of raw Anthropic stream events into
 * LangChain `ChatModelStreamEvent`s with typed deltas.
 */
export async function* convertAnthropicStream(
  source: AsyncIterable<AnthropicStreamInputEvent>,
  options: ConvertAnthropicStreamOptions = {}
): AsyncGenerator<ChatModelStreamEvent> {
  const shouldStreamUsage = options.streamUsage ?? true;

  // Track accumulated state per content block (for finalization)
  const blockAccumulators = new Map<number, BlockAccumulator>();
  let usageSnapshot: UsageMetadata | undefined;
  let eventUsage: AnthropicEventUsage | undefined;
  let stopReason: string | null = null;
  let container: Extract<
    AnthropicMessageStreamEvent,
    { type: 'message_start' }
  >['message']['container'] = null;

  for await (const data of source) {
    switch (data.type) {
    // ── Message lifecycle ──────────────────────────────────
    case 'message_start': {
      const { usage, id, model } = data.message;
      container = data.message.container;
      if (shouldStreamUsage) {
        eventUsage = {
          inputTokens: usage.input_tokens,
          cacheCreationInputTokens: usage.cache_creation_input_tokens ?? 0,
          cacheReadInputTokens: usage.cache_read_input_tokens ?? 0,
          outputTokens: usage.output_tokens,
        };
        usageSnapshot = buildUsageSnapshot(eventUsage);
      }
      yield {
        event: 'message-start' as const,
        id,
        ...(usageSnapshot ? { usage: usageSnapshot } : {}),
      };
      yield {
        event: 'provider' as const,
        provider: 'anthropic',
        name: 'message_start',
        payload: { model, id },
      };
      break;
    }

    case 'message_delta': {
      stopReason = data.delta.stop_reason;
      container = data.delta.container ?? container;
      if (shouldStreamUsage) {
        eventUsage = updateEventUsage(eventUsage, data.usage);
        usageSnapshot = buildUsageSnapshot(eventUsage);
        yield { event: 'usage' as const, usage: usageSnapshot };
      }
      if (
        'context_management' in data.delta &&
          data.delta.context_management != null
      ) {
        yield {
          event: 'provider' as const,
          provider: 'anthropic',
          name: 'context_management',
          payload: data.delta.context_management,
        };
      }
      break;
    }

    case 'message_stop': {
      const finishEvent = {
        event: 'message-finish' as const,
        reason: mapStopReason(stopReason),
        ...(usageSnapshot ? { usage: usageSnapshot } : {}),
        responseMetadata: {
          model_provider: 'anthropic',
          ...(container != null ? { container } : {}),
        },
      };
      yield finishEvent;
      break;
    }

    case 'error': {
      const streamError = new Error(data.error.message);
      streamError.name = data.error.type;
      throw streamError;
    }

    // ── Content block lifecycle ───────────────────────────
    case 'content_block_start': {
      const { index, content_block } = data;
      const mapped = mapBlockToContentBlock(content_block, index);
      blockAccumulators.set(index, { ...mapped });
      yield {
        event: 'content-block-start' as const,
        index,
        content: mapped as unknown as ContentBlock,
      };
      break;
    }

    case 'content_block_delta': {
      const { index, delta } = data;
      const acc = blockAccumulators.get(index);
      if (!acc) break;

      const { contentDelta, accumulated } = applyAnthropicDelta(acc, delta);
      blockAccumulators.set(index, accumulated);

      yield {
        event: 'content-block-delta' as const,
        index,
        delta: contentDelta,
      };
      break;
    }

    case 'content_block_stop': {
      const { index } = data;
      const acc = blockAccumulators.get(index);
      if (!acc) break;

      const finalized = finalizeBlock(acc);
      yield {
        event: 'content-block-finish' as const,
        index,
        content: finalized,
      };
      blockAccumulators.delete(index);
      break;
    }

    // ── Unhandled → provider passthrough ───────────────────
    default: {
      const providerData = data as AnthropicMessageStreamEvent;
      yield {
        event: 'provider' as const,
        provider: 'anthropic',
        name: providerData.type,
        payload: providerData,
      };
      break;
    }
    }
  }
}

// ─── Internal helpers ───────────────────────────────────────────

function mapStopReason(stopReason: string | null | undefined): FinishReason {
  switch (stopReason) {
  case 'end_turn':
  case 'stop_sequence':
    return 'stop';
  case 'tool_use':
    return 'tool_use';
  case 'max_tokens':
  case 'model_context_window_exceeded':
    return 'length';
  case 'refusal':
    return 'content_filter';
  default:
    return 'stop';
  }
}

function updateEventUsage(
  previous: AnthropicEventUsage | undefined,
  current: AnthropicUsageData
): AnthropicEventUsage {
  return {
    inputTokens: getCumulativeUsageValue(
      current.input_tokens,
      previous?.inputTokens
    ),
    cacheCreationInputTokens: getCumulativeUsageValue(
      current.cache_creation_input_tokens,
      previous?.cacheCreationInputTokens
    ),
    cacheReadInputTokens: getCumulativeUsageValue(
      current.cache_read_input_tokens,
      previous?.cacheReadInputTokens
    ),
    outputTokens: getCumulativeUsageValue(
      current.output_tokens,
      previous?.outputTokens
    ),
  };
}

function getCumulativeUsageValue(
  current: number | null | undefined,
  previous: number | undefined
): number {
  return Math.max(previous ?? 0, current ?? 0);
}

function buildUsageSnapshot(usage: AnthropicEventUsage): UsageMetadata {
  const metadata = getAnthropicUsageMetadata({
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    cache_creation_input_tokens: usage.cacheCreationInputTokens,
    cache_read_input_tokens: usage.cacheReadInputTokens,
  });
  if (metadata == null) {
    throw new Error('Anthropic usage metadata was not created');
  }
  return metadata;
}

function getStringField(record: BlockAccumulator, key: string): string {
  const value = record[key];
  return typeof value === 'string' ? value : '';
}

function getArrayField(record: BlockAccumulator, key: string): unknown[] {
  const value = record[key];
  return Array.isArray(value) ? value : [];
}

function mapBlockToContentBlock(
  block: AnthropicContentBlock,
  index: number
): BlockAccumulator {
  switch (block.type) {
  case 'text':
    return {
      type: 'text' as const,
      text: block.text,
      ...(block.citations != null ? { citations: block.citations } : {}),
      index,
    };
  case 'thinking':
    return {
      type: 'reasoning' as const,
      reasoning: block.thinking,
      index,
    };
  case 'redacted_thinking':
    return { ...block, index };
  case 'tool_use':
  case 'server_tool_use':
    return {
      type: block.type === 'tool_use' ? 'tool_call_chunk' : 'server_tool_call_chunk',
      id: block.id,
      name: block.name,
      ...('caller' in block ? { caller: block.caller } : {}),
      args: '',
      index,
    };
  case 'compaction':
    return { ...block, index };
  default:
    return isAnthropicServerToolResultType(block.type)
      ? { ...block, index }
      : { type: 'non_standard' as const, value: { ...block }, index };
  }
}

/**
 * Map an Anthropic content_block_delta to a content block delta
 * and update the accumulated state.
 */
function applyAnthropicDelta(
  accumulated: BlockAccumulator,
  delta: AnthropicContentDelta
): {
  contentDelta: ContentBlockDelta;
  accumulated: BlockAccumulator;
} {
  const rawDelta = delta as unknown as Record<string, unknown>;
  switch (delta.type) {
  case 'text_delta':
    return {
      contentDelta: { type: 'text-delta' as const, text: delta.text },
      accumulated: {
        ...accumulated,
        text: getStringField(accumulated, 'text') + delta.text,
      },
    };

  case 'thinking_delta':
    return {
      contentDelta: {
        type: 'reasoning-delta' as const,
        reasoning: delta.thinking,
      },
      accumulated: {
        ...accumulated,
        reasoning: getStringField(accumulated, 'reasoning') + delta.thinking,
      },
    };

  case 'input_json_delta': {
    const newArgs = getStringField(accumulated, 'args') + delta.partial_json;
    return {
      contentDelta: {
        type: 'block-delta' as const,
        fields: {
          type: getStringField(accumulated, 'type'),
          args: newArgs,
        },
      },
      accumulated: { ...accumulated, args: newArgs },
    };
  }

  case 'citations_delta': {
    const citations = [
      ...getArrayField(accumulated, 'citations'),
      delta.citation,
    ];
    return {
      contentDelta: {
        type: 'block-delta' as const,
        fields: {
          type: getStringField(accumulated, 'type'),
          citations,
        },
      },
      accumulated: {
        ...accumulated,
        citations,
      },
    };
  }

  case 'signature_delta':
    return {
      contentDelta: {
        type: 'block-delta' as const,
        fields: {
          type: getStringField(accumulated, 'type'),
          signature: delta.signature,
        },
      },
      accumulated: { ...accumulated, signature: delta.signature },
    };

  case 'compaction_delta': {
    const previousContent = accumulated.content;
    const content =
        delta.content == null
          ? (previousContent ?? null)
          : getStringField(accumulated, 'content') + delta.content;
    return {
      contentDelta: {
        type: 'block-delta' as const,
        fields: {
          type: 'compaction',
          content,
          encrypted_content: delta.encrypted_content,
        },
      },
      accumulated: {
        ...accumulated,
        content,
        encrypted_content: delta.encrypted_content,
      },
    };
  }

  default:
    return {
      contentDelta: {
        type: 'block-delta' as const,
        fields: {
          type: getStringField(accumulated, 'type'),
          ...rawDelta,
        },
      },
      accumulated,
    };
  }
}

function finalizeBlock(accumulated: BlockAccumulator): ContentBlock {
  if (
    accumulated.type === 'tool_call_chunk' ||
    accumulated.type === 'server_tool_call_chunk'
  ) {
    const finalType =
      accumulated.type === 'tool_call_chunk'
        ? ('tool_call' as const)
        : ('server_tool_call' as const);
    const args = getStringField(accumulated, 'args');
    let parsedArgs: unknown;
    try {
      parsedArgs = JSON.parse(args || '{}');
    } catch {
      return {
        type: 'invalid_tool_call' as const,
        id: getStringField(accumulated, 'id'),
        name: getStringField(accumulated, 'name'),
        args,
        error: 'Failed to parse tool call arguments as JSON',
      } as ContentBlock.Tools.InvalidToolCall;
    }
    return {
      type: finalType,
      id: getStringField(accumulated, 'id'),
      name: getStringField(accumulated, 'name'),
      args: parsedArgs,
      ...(accumulated.caller != null ? { caller: accumulated.caller } : {}),
    } as ContentBlock;
  }

  const { index: _index, ...rest } = accumulated;
  return rest as ContentBlock;
}

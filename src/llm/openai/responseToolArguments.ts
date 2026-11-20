import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { convertResponsesDeltaToChatGenerationChunk } from '@langchain/openai';
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type { OpenAIClient } from '@langchain/openai';

type ResponseEvent = OpenAIClient.Responses.ResponseStreamEvent;
type ResponseConverter = (event: ResponseEvent) => ChatGenerationChunk | null;
type FunctionCall = OpenAIClient.Responses.ResponseFunctionToolCall;
type FunctionCallSnapshot = Pick<FunctionCall, 'arguments'> &
  Partial<Pick<FunctionCall, 'id' | 'call_id' | 'name'>>;

function reconcileArguments(
  calls: Map<number, ToolCallChunk>,
  index: number,
  snapshot: FunctionCallSnapshot
): ChatGenerationChunk | null {
  const call = calls.get(index);
  const previousArgs = call?.args ?? '';
  if (
    !snapshot.arguments.startsWith(previousArgs) ||
    (call?.name != null &&
      snapshot.name != null &&
      call.name !== snapshot.name) ||
    (call?.id != null &&
      snapshot.call_id != null &&
      call.id !== snapshot.call_id)
  ) {
    throw new Error(
      `Responses function call at output index ${index} changed after streaming.`
    );
  }
  const missingArgs = snapshot.arguments.slice(previousArgs.length);
  const missingId = call?.id == null ? snapshot.call_id : undefined;
  const missingName = call?.name == null ? snapshot.name : undefined;
  if (missingArgs === '' && missingId == null && missingName == null) {
    return null;
  }
  calls.set(index, {
    type: 'tool_call_chunk',
    index,
    id: call?.id ?? snapshot.call_id,
    name: call?.name ?? snapshot.name,
    args: snapshot.arguments,
  });
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          type: 'tool_call_chunk',
          index,
          args: missingArgs,
          ...(missingId != null && { id: missingId }),
          ...(missingName != null && { name: missingName }),
        },
      ],
      additional_kwargs:
        missingId != null && snapshot.id != null
          ? {
            __openai_function_call_ids__: { [missingId]: snapshot.id },
          }
          : {},
      response_metadata: { model_provider: 'openai' },
    }),
  });
}

/** Reconcile terminal snapshots with the append-only tool-call chunk stream. */
export function createResponsesToolArgumentConverter(): ResponseConverter {
  const calls = new Map<number, ToolCallChunk>();

  return (event): ChatGenerationChunk | null => {
    if (
      event.type === 'response.output_item.added' &&
      event.item.type === 'function_call'
    ) {
      calls.set(event.output_index, {
        type: 'tool_call_chunk',
        index: event.output_index,
        id: event.item.call_id,
        name: event.item.name,
        args: event.item.arguments,
      });
    } else if (event.type === 'response.function_call_arguments.delta') {
      const call = calls.get(event.output_index) ?? {
        type: 'tool_call_chunk',
        index: event.output_index,
        args: '',
      };
      call.args = (call.args ?? '') + event.delta;
      calls.set(event.output_index, call);
    } else if (event.type === 'response.function_call_arguments.done') {
      return reconcileArguments(calls, event.output_index, event);
    } else if (
      event.type === 'response.output_item.done' &&
      event.item.type === 'function_call'
    ) {
      return reconcileArguments(calls, event.output_index, event.item);
    }
    return convertResponsesDeltaToChatGenerationChunk(event);
  };
}

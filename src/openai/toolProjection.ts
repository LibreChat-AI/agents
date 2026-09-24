import type {
  OpenAIChatCompletionChunkChoice,
  OpenAIToolCall,
  OpenAIStreamTracker,
} from './index';
import type { EventHandler, ModelResponseEvent } from '@/types';
import { GraphEvents } from '@/common';
import { serializeToolArguments } from './arguments';

interface OpenAIToolCallStreamOptions {
  /** Synchronous framing only. Async transport/backpressure is a separate boundary. */
  emit?: (delta: OpenAIChatCompletionChunkChoice['delta']) => void;
  signal?: AbortSignal;
  /** Bound retained complete-before-publish output, not model execution. Default: 1024. */
  maxToolCalls?: number;
  /** UTF-8 bytes of serialized arguments, names and IDs retained across accepted responses. Default: 4 MiB. */
  maxBufferedBytes?: number;
}

/** Streaming hosts share the finalizer's tracker; map-only hosts collect JSON output. */
export type OpenAIToolCallStreamConfig = OpenAIToolCallStreamOptions &
  (
    | {
        tracker: OpenAIStreamTracker;
        toolCalls?: never;
        emit: NonNullable<OpenAIToolCallStreamOptions['emit']>;
      }
    | { toolCalls: Map<number, OpenAIToolCall>; tracker?: never }
  );

export interface OpenAIToolCallStream {
  /** Pass directly to Run.create({ customHandlers: stream.handlers }). */
  handlers: Record<string, EventHandler>;
  /** Publish only after the host verifies that the entire run completed successfully. */
  finish: () => void;
  abort: () => void;
}

function positiveLimit(value: number | undefined, fallback: number): number {
  const limit = value ?? fallback;
  if (!Number.isSafeInteger(limit) || limit <= 0) {
    throw new Error('Tool projection limits must be positive safe integers');
  }
  return limit;
}

/** Serializes finalized, accepted tool calls. No provider fragments or attempt inference. */
export function createOpenAIToolCallStream(
  config: OpenAIToolCallStreamConfig
): OpenAIToolCallStream {
  const { emit, signal, tracker } = config;
  const suppliedMap: unknown = config.toolCalls;
  const suppliedEmitter: unknown = emit;
  if (tracker != null && typeof suppliedEmitter !== 'function') {
    throw new Error('A streaming tracker requires an emitter');
  }
  if (tracker != null && suppliedMap != null) {
    throw new Error('Provide a tracker or a tool-call map, not both');
  }
  const toolCalls = tracker?.toolCalls ?? config.toolCalls;
  if (toolCalls == null)
    throw new Error('Provide a tracker or a tool-call map');
  let previousChunkKind: OpenAIStreamTracker['lastChunkKind'];
  const maxCalls = positiveLimit(config.maxToolCalls, 1024);
  const maxBytes = positiveLimit(config.maxBufferedBytes, 4 * 1024 * 1024);
  const calls: Array<{ id: string; name: string; arguments: string }> = [];
  const providerIds = new Set<string>();
  const acceptedIds = new Set<string>();
  let bufferedBytes = 0;
  let phase: 'open' | 'emitting' | 'finished' | 'aborted' = 'open';

  const release = (): void => {
    calls.length = 0;
    acceptedIds.clear();
    providerIds.clear();
    bufferedBytes = 0;
  };
  const abort = (): void => {
    const phaseBeforeAbort = phase;
    if (phase !== 'finished') {
      phase = 'aborted';
      toolCalls.clear();
      if (tracker != null && phaseBeforeAbort === 'emitting')
        tracker.lastChunkKind = previousChunkKind;
    }
    release();
  };
  const checkCancellation = (): void => {
    if (signal?.aborted === true) abort();
    if (phase === 'aborted') {
      const error = new Error('Agent response aborted');
      error.name = 'AbortError';
      throw error;
    }
  };
  const emitDelta = (delta: OpenAIChatCompletionChunkChoice['delta']): void => {
    const result: unknown = emit?.(delta);
    if (
      result != null &&
      typeof result === 'object' &&
      'then' in result &&
      typeof result.then === 'function'
    ) {
      // Observe a mistakenly async sink's rejection, then fail closed. Never silently
      // publish a successful terminal response before pending writes settle.
      void Promise.resolve(result).catch(() => undefined);
      throw new Error('Tool projection requires a synchronous emitter');
    }
  };
  const accept = (result: ModelResponseEvent): void => {
    if (phase !== 'open') return;
    try {
      checkCancellation();
      if (result.invalidToolCalls.length > 0) {
        throw new Error('Accepted model response contains invalid tool calls');
      }
      if (result.toolCalls.length === 0) return;
      if (result.id.trim() === '' || acceptedIds.has(result.id)) {
        throw new Error('Missing or repeated accepted model response identity');
      }
      if (calls.length + result.toolCalls.length > maxCalls) {
        throw new Error('Tool projection call limit exceeded');
      }
      acceptedIds.add(result.id);
      for (const call of result.toolCalls) {
        if (typeof call.name !== 'string' || call.name.trim() === '') {
          throw new Error('Accepted tool call is missing its name');
        }
        const id = call.id ?? '';
        if (typeof id !== 'string')
          throw new Error('Accepted tool call ID must be a string');
        if (id !== '') providerIds.add(id);
        const identityBytes =
          Buffer.byteLength(id, 'utf8') + Buffer.byteLength(call.name, 'utf8');
        const args = serializeToolArguments(
          call.args,
          maxBytes - bufferedBytes - identityBytes
        );
        bufferedBytes += identityBytes + Buffer.byteLength(args, 'utf8');
        calls.push({ id, name: call.name, arguments: args });
      }
    } catch (error) {
      abort();
      throw error;
    }
  };

  return {
    handlers: {
      [GraphEvents.ON_MODEL_RESPONSE]: {
        handle: (event, data): void => {
          if (
            event !== GraphEvents.ON_MODEL_RESPONSE ||
            data == null ||
            !('type' in data) ||
            data.type !== 'model_response'
          )
            return;
          accept(data);
        },
      },
    },
    abort,
    finish: (): void => {
      if (phase === 'finished' || phase === 'emitting') return;
      checkCancellation();
      previousChunkKind = tracker?.lastChunkKind;
      phase = 'emitting';
      try {
        // Do not allocate synthetic IDs until every accepted response has arrived.
        // This reserves provider IDs even when they occur in a later invocation.
        const usedIds = new Set<string>();
        const ready: OpenAIToolCall[] = [];
        let reservedBytes = bufferedBytes;
        let nextSyntheticId = 0;
        for (const call of calls) {
          let id = call.id;
          if (id === '' || usedIds.has(id)) {
            do {
              id = `call_${nextSyntheticId++}`;
            } while (providerIds.has(id) || usedIds.has(id));
            reservedBytes +=
              Buffer.byteLength(id, 'utf8') -
              Buffer.byteLength(call.id, 'utf8');
            if (reservedBytes > maxBytes)
              throw new Error('Tool projection buffer limit exceeded');
          }
          usedIds.add(id);
          ready.push(
            Object.freeze({
              id,
              type: 'function',
              function: Object.freeze({
                name: call.name,
                arguments: call.arguments,
              }),
            })
          );
        }
        if (tracker != null && ready.length > 0 && !tracker.hasRole) {
          tracker.hasRole = true;
          emitDelta({ role: 'assistant' });
          checkCancellation();
        }
        for (let index = 0; index < ready.length; index++) {
          checkCancellation();
          const call = ready[index];
          toolCalls.set(index, call);
          if (tracker != null) tracker.lastChunkKind = 'tool_call';
          emitDelta({
            tool_calls: [
              {
                index,
                id: call.id,
                type: 'function',
                function: { name: call.function.name, arguments: '' },
              },
            ],
          });
          checkCancellation();
          emitDelta({
            tool_calls: [
              { index, function: { arguments: call.function.arguments } },
            ],
          });
          checkCancellation();
        }
        phase = 'finished';
      } catch (error) {
        abort();
        throw error;
      } finally {
        release();
      }
    },
  };
}

import type { OpenAIChatCompletionChunkChoice, OpenAIToolCall } from './index';
import type { EventHandler, ModelResponseEvent } from '@/types';
import { GraphEvents } from '@/common';

export interface OpenAIToolCallStreamConfig {
  toolCalls: Map<number, OpenAIToolCall>;
  /** Synchronous framing only. Async transport/backpressure is a separate boundary. */
  emit?: (delta: OpenAIChatCompletionChunkChoice['delta']) => void;
  signal?: AbortSignal;
  /** Bound retained complete-before-publish output, not model execution. Default: 1024. */
  maxToolCalls?: number;
  /** UTF-8 bytes of serialized arguments, names and IDs retained across accepted responses. Default: 4 MiB. */
  maxBufferedBytes?: number;
}

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
  const { toolCalls, emit, signal } = config;
  const maxCalls = positiveLimit(config.maxToolCalls, 1024);
  const maxBytes = positiveLimit(config.maxBufferedBytes, 4 * 1024 * 1024);
  const calls: OpenAIToolCall[] = [];
  const acceptedIds = new Set<string>();
  const outwardIds = new Set<string>();
  let bufferedBytes = 0;
  let nextSyntheticId = 0;
  let phase: 'open' | 'emitting' | 'finished' | 'aborted' = 'open';

  const release = (): void => {
    calls.length = 0;
    acceptedIds.clear();
    outwardIds.clear();
    bufferedBytes = 0;
  };
  const abort = (): void => {
    if (phase !== 'finished') {
      phase = 'aborted';
      toolCalls.clear();
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
        const finalArgs: unknown = call.args;
        if (
          finalArgs == null ||
          typeof finalArgs !== 'object' ||
          Array.isArray(finalArgs)
        ) {
          throw new Error('Accepted tool call arguments must be an object');
        }
        let args: string;
        try {
          const encoded: unknown = JSON.stringify(
            finalArgs,
            (_key, value: unknown): unknown => {
              // Do not silently change executed arguments (e.g. NaN to null, or omit undefined).
              if (
                value === undefined ||
                typeof value === 'function' ||
                typeof value === 'symbol' ||
                typeof value === 'bigint' ||
                (typeof value === 'number' && !Number.isFinite(value))
              ) {
                throw new Error();
              }
              return value;
            }
          );
          if (typeof encoded !== 'string' || encoded[0] !== '{')
            throw new Error();
          args = encoded;
        } catch {
          throw new Error(
            'Accepted tool call arguments are not JSON serializable'
          );
        }
        let id = call.id ?? '';
        if (id === '' || outwardIds.has(id)) {
          // Each collision allocates at most one new candidate per existing ID.
          do {
            id = `call_${nextSyntheticId++}`;
          } while (outwardIds.has(id));
        }
        bufferedBytes +=
          Buffer.byteLength(args, 'utf8') +
          Buffer.byteLength(id, 'utf8') +
          Buffer.byteLength(call.name, 'utf8');
        if (bufferedBytes > maxBytes)
          throw new Error('Tool projection buffer limit exceeded');
        outwardIds.add(id);
        calls.push(
          Object.freeze({
            id,
            type: 'function',
            function: Object.freeze({ name: call.name, arguments: args }),
          })
        );
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
      phase = 'emitting';
      try {
        for (let index = 0; index < calls.length; index++) {
          checkCancellation();
          const call = calls[index];
          toolCalls.set(index, call);
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

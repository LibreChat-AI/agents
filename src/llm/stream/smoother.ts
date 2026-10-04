import { isProtectedProviderTextSignal } from '@/protection/providerText';

export const DEFAULT_STREAM_DELAY = 25;
export const SMOOTH_TARGET_LATENCY_MS = 250;
export const MAX_STREAM_QUEUE_CHUNKS = 256;
export const MAX_STREAM_QUEUE_TEXT_CHARS = 8192;
export const MAX_SMOOTH_ITEM_SEGMENT_CHARS = 4096;
export const STREAM_CHUNK_MIN_SIZE = 4;
export const STREAM_BOUNDARIES: ReadonlySet<string> = new Set([
  ' ',
  '.',
  ',',
  '!',
  '?',
  ';',
  ':',
]);

export const STREAM_ABORT_MESSAGE = 'AbortError: User aborted the request.';
export const STREAM_PRODUCER_FAILURE = 'Stream producer failed.';

/**
 * How long generator teardown waits for the background producer to observe a
 * consumer close before abandoning it. Well-behaved streams settle in
 * microseconds (the next enqueue throws); a stalled provider that ignores
 * aborts otherwise blocks teardown — and abort propagation — indefinitely.
 * An abandoned producer still self-terminates on its next enqueue attempt.
 */
export const PRODUCER_CLOSE_GRACE_MS = 1000;

/**
 * Resolves a configured stream delay to its effective value (default 25ms;
 * 0 disables smoothing). Non-finite inputs (NaN from a malformed config
 * value, ±Infinity) normalize to the default rather than poisoning piece
 * arithmetic downstream.
 */
export function resolveStreamDelay(delay?: number): number {
  if (delay == null || !Number.isFinite(delay)) {
    return DEFAULT_STREAM_DELAY;
  }
  return Math.max(0, delay);
}

export function isSignalAborted(signal?: AbortSignal): boolean {
  return signal?.aborted === true;
}

/**
 * How far past the target size the word-boundary search may extend before
 * hard-cutting. Natural language hits a boundary within a few characters;
 * boundary-free runs (base64, minified data, long identifiers) must not
 * stretch a piece — or an admission segment — arbitrarily far past its
 * budget.
 */
export const STREAM_BOUNDARY_LOOKAHEAD_CHARS = 64;

export function findStreamChunkBoundary(
  text: string,
  minSize: number
): number {
  if (minSize >= text.length) {
    return text.length;
  }

  const scanEnd = Math.min(
    text.length,
    minSize + STREAM_BOUNDARY_LOOKAHEAD_CHARS
  );
  for (let position = minSize; position < scanEnd; position++) {
    if (STREAM_BOUNDARIES.has(text[position])) {
      return position + 1;
    }
  }

  return scanEnd;
}

/**
 * Backlog-proportional piece sizing: emit enough per tick that the current
 * backlog drains in ~`targetLatencyMs`, so render lag stays pinned near the
 * target regardless of how fast the provider streams. Token-sized arrivals
 * never exceed the minimum piece, matching the legacy fixed-size splitter.
 */
export function computeAdaptivePieceSize(
  bufferedTextLength: number,
  tickMs: number,
  targetLatencyMs: number = SMOOTH_TARGET_LATENCY_MS
): number {
  if (bufferedTextLength <= 0) {
    return STREAM_CHUNK_MIN_SIZE;
  }
  if (tickMs <= 0 || targetLatencyMs <= 0) {
    return bufferedTextLength;
  }
  return Math.max(
    STREAM_CHUNK_MIN_SIZE,
    Math.ceil((bufferedTextLength * tickMs) / targetLatencyMs)
  );
}

/**
 * A cadence, not an additive sleep: time the consumer already spent since the
 * last visible emission counts against the target delay, so slow downstream
 * handlers never compound latency.
 */
export function getCadencedStreamDelay({
  targetDelay,
  lastVisibleTextAt,
  now,
}: {
  targetDelay: number;
  lastVisibleTextAt?: number;
  now: number;
}): number {
  if (targetDelay <= 0 || lastVisibleTextAt == null) {
    return 0;
  }
  return Math.max(0, targetDelay - (now - lastVisibleTextAt));
}

/** Abort-aware sleep that resolves (never rejects) on abort; callers re-check the signal. */
export async function waitForStreamDelay(
  delay: number,
  signal?: AbortSignal
): Promise<void> {
  if (delay <= 0 || isSignalAborted(signal)) {
    return;
  }
  await new Promise<void>((resolve) => {
    const timeoutRef: { current?: ReturnType<typeof setTimeout> } = {};
    const onAbort = (): void => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      signal?.removeEventListener('abort', onAbort);
      resolve();
    };
    timeoutRef.current = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, delay);
    signal?.addEventListener('abort', onAbort, { once: true });
    if (isSignalAborted(signal)) {
      onAbort();
    }
  });
}

export type SmoothPiece = {
  text: string;
  isFirst: boolean;
  isLast: boolean;
};

/**
 * One classified unit of provider stream output.
 *
 * - `smooth: true` — visible text, paced at the configured cadence and (unless
 *   `atomic`) sliced adaptively at dequeue time.
 * - `atomic: true` — paced as a single piece, never split (text-bearing chunks
 *   whose metadata cannot survive slicing, e.g. logprobs / finish_reason).
 * - `smooth: false` — passthrough: tool-call deltas, usage-only, id-only and
 *   seal chunks. Zero delay, strict FIFO with the text around them.
 *
 * `emit` builds the provider-specific output for one piece; `isFirst` lets
 * providers keep usage_metadata on only the first piece of a split.
 */
export type SmoothItem<TEmit> = {
  text: string;
  smooth: boolean;
  atomic?: boolean;
  emit: (piece: SmoothPiece) => TEmit;
};

type ProducerState = {
  done: boolean;
  failed: boolean;
  error?: unknown;
};

type QueuedSmoothItem<TEmit> = {
  item: SmoothItem<TEmit>;
  textLength: number;
};

/**
 * Bounded producer/consumer smoothing engine.
 *
 * The producer drains `source` eagerly into a bounded queue (the buffer is the
 * backlog measurement adaptive sizing needs); at capacity it parks, applying
 * backpressure to the underlying stream. The consumer emits paced pieces,
 * decrementing the text budget and waking the producer *before* each cadenced
 * sleep so the provider stream keeps being read during pacing.
 *
 * `delayMs <= 0` disables smoothing entirely: every item passes through FIFO,
 * unsplit and undelayed.
 */
export async function* smoothStream<TEmit>({
  source,
  delayMs,
  signal,
  abortUpstream,
}: {
  source: AsyncIterable<SmoothItem<TEmit>>;
  delayMs: number;
  signal?: AbortSignal;
  abortUpstream?: () => void;
}): AsyncGenerator<TEmit> {
  if (!(delayMs > 0) || isProtectedProviderTextSignal(signal)) {
    /** Disabled/protected smoothing preserves fully lazy streaming: no background
     * producer, no read-ahead — each provider chunk is pulled only when the
     * consumer asks, exactly like the pre-engine pass-through paths. */
    for await (const item of source) {
      if (isSignalAborted(signal)) {
        abortUpstream?.();
        throw new Error(STREAM_ABORT_MESSAGE);
      }
      yield item.emit({ text: item.text, isFirst: true, isLast: true });
    }
    return;
  }

  const queuedItems: QueuedSmoothItem<TEmit>[] = [];
  const producerState: ProducerState = { done: false, failed: false };
  let queuedItemIndex = 0;
  let bufferedTextLength = 0;
  let consumerClosed = false;
  let notifyConsumer: (() => void) | undefined;
  let notifyProducer: (() => void) | undefined;

  const notifyConsumerForItem = (): void => {
    notifyConsumer?.();
    notifyConsumer = undefined;
  };

  const notifyProducerForSpace = (): void => {
    notifyProducer?.();
    notifyProducer = undefined;
  };

  const hasQueuedItems = (): boolean => queuedItemIndex < queuedItems.length;

  const getQueuedItemCount = (): number =>
    queuedItems.length - queuedItemIndex;

  const isQueueAtCapacity = (): boolean =>
    getQueuedItemCount() >= MAX_STREAM_QUEUE_CHUNKS ||
    bufferedTextLength >= MAX_STREAM_QUEUE_TEXT_CHARS;

  /** Abort-aware: a consumer parked on an empty queue must wake when the
   * signal fires even if the provider stream never honors the abort — the
   * loop's top-of-iteration check then throws the canonical error. */
  const waitForNextItem = async (): Promise<void> => {
    if (
      hasQueuedItems() ||
      producerState.done ||
      producerState.failed ||
      isSignalAborted(signal)
    ) {
      return;
    }
    await new Promise<void>((resolve) => {
      const onAbort = (): void => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      };
      notifyConsumer = (): void => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      if (isSignalAborted(signal)) {
        onAbort();
      }
    });
  };

  const waitForQueueSpace = async (): Promise<void> => {
    while (
      isQueueAtCapacity() &&
      !consumerClosed &&
      !isSignalAborted(signal)
    ) {
      await new Promise<void>((resolve) => {
        const onAbort = (): void => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        };
        const onSpace = (): void => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        };
        notifyProducer = onSpace;
        signal?.addEventListener('abort', onAbort, { once: true });
        if (isSignalAborted(signal)) {
          onAbort();
        }
      });
    }
  };

  const dequeue = (): QueuedSmoothItem<TEmit> | undefined => {
    if (!hasQueuedItems()) {
      return undefined;
    }
    const queuedItem = queuedItems[queuedItemIndex];
    queuedItemIndex++;
    if (queuedItemIndex > 128 && queuedItemIndex * 2 >= queuedItems.length) {
      queuedItems.splice(0, queuedItemIndex);
      queuedItemIndex = 0;
    }
    return queuedItem;
  };

  const throwAborted = (): never => {
    abortUpstream?.();
    throw new Error(STREAM_ABORT_MESSAGE);
  };

  const enqueue = async (item: SmoothItem<TEmit>): Promise<void> => {
    await waitForQueueSpace();
    if (consumerClosed || isSignalAborted(signal)) {
      throwAborted();
    }
    const textLength = item.smooth ? item.text.length : 0;
    queuedItems.push({ item, textLength });
    bufferedTextLength += textLength;
    notifyConsumerForItem();
  };

  /**
   * Oversized splittable items are segmented at admission so a single giant
   * provider chunk cannot blow past the text budget: each segment re-checks
   * capacity, so the producer parks mid-chunk once the buffer fills — the
   * same bound the legacy split-before-enqueue queues enforced. The wrapped
   * emit maps segment-local pieces back to chunk-global isFirst/isLast so
   * provider clone contracts are unaffected.
   */
  const enqueueSegmented = async (item: SmoothItem<TEmit>): Promise<void> => {
    if (
      !item.smooth ||
      item.atomic === true ||
      item.text.length <= MAX_SMOOTH_ITEM_SEGMENT_CHARS
    ) {
      await enqueue(item);
      return;
    }

    const segments: { start: number; end: number }[] = [];
    let offset = 0;
    while (offset < item.text.length) {
      const end =
        offset +
        findStreamChunkBoundary(
          item.text.slice(offset),
          MAX_SMOOTH_ITEM_SEGMENT_CHARS
        );
      segments.push({ start: offset, end });
      offset = end;
    }

    for (let i = 0; i < segments.length; i++) {
      const isFirstSegment = i === 0;
      const isLastSegment = i === segments.length - 1;
      await enqueue({
        text: item.text.slice(segments[i].start, segments[i].end),
        smooth: true,
        emit: (piece) =>
          item.emit({
            text: piece.text,
            isFirst: isFirstSegment && piece.isFirst,
            isLast: isLastSegment && piece.isLast,
          }),
      });
    }
  };

  const producer = (async (): Promise<void> => {
    try {
      for await (const item of source) {
        if (isSignalAborted(signal)) {
          throwAborted();
        }
        await enqueueSegmented(item);
      }
    } catch (error) {
      producerState.failed = true;
      producerState.error = error;
    } finally {
      producerState.done = true;
      notifyConsumerForItem();
    }
  })();

  let hasEmittedText = false;
  let lastVisibleTextAt: number | undefined;
  let drainTicksRemaining: number | undefined;
  let current: QueuedSmoothItem<TEmit> | undefined;
  let headOffset = 0;
  let keepStreaming = true;
  try {
    while (keepStreaming) {
      if (isSignalAborted(signal)) {
        throwAborted();
      }

      if (current == null) {
        await waitForNextItem();
        current = dequeue();
        headOffset = 0;
      }

      if (current == null) {
        if (producerState.failed) {
          throw producerState.error ?? new Error(STREAM_PRODUCER_FAILURE);
        }
        if (producerState.done) {
          keepStreaming = false;
        }
        continue;
      }

      const { item } = current;

      if (!item.smooth) {
        notifyProducerForSpace();
        current = undefined;
        yield item.emit({ text: item.text, isFirst: true, isLast: true });
        continue;
      }

      if (item.text === '') {
        bufferedTextLength = Math.max(
          0,
          bufferedTextLength - current.textLength
        );
        notifyProducerForSpace();
        current = undefined;
        continue;
      }

      /** Once the producer is done the backlog is final: drain it linearly
       * across the remaining target window instead of letting the
       * proportional formula decay geometrically and stretch the tail. */
      if (producerState.done && drainTicksRemaining == null) {
        drainTicksRemaining = Math.max(
          1,
          Math.floor(SMOOTH_TARGET_LATENCY_MS / delayMs)
        );
      }
      const tickBudget =
        drainTicksRemaining != null
          ? Math.max(
            STREAM_CHUNK_MIN_SIZE,
            Math.ceil(bufferedTextLength / drainTicksRemaining)
          )
          : computeAdaptivePieceSize(bufferedTextLength, delayMs);
      if (drainTicksRemaining != null && drainTicksRemaining > 1) {
        drainTicksRemaining -= 1;
      }

      await waitForStreamDelay(
        getCadencedStreamDelay({
          targetDelay: hasEmittedText ? delayMs : 0,
          lastVisibleTextAt,
          now: Date.now(),
        }),
        signal
      );
      if (isSignalAborted(signal)) {
        throwAborted();
      }
      hasEmittedText = true;
      lastVisibleTextAt = Date.now();

      if (item.atomic === true) {
        bufferedTextLength = Math.max(
          0,
          bufferedTextLength - current.textLength
        );
        notifyProducerForSpace();
        current = undefined;
        yield item.emit({ text: item.text, isFirst: true, isLast: true });
        continue;
      }

      /** One cadence tick drains up to the adaptive budget ACROSS queued
       * items, so token-sized provider deltas coalesce instead of costing a
       * full tick each; passthrough items flush free mid-batch (FIFO), and
       * atomic items end the batch to take their own tick. */
      let consumed = 0;
      while (consumed < tickBudget) {
        if (isSignalAborted(signal)) {
          throwAborted();
        }
        if (current == null) {
          if (!hasQueuedItems()) {
            break;
          }
          current = dequeue();
          headOffset = 0;
          if (current == null) {
            break;
          }
        }

        const batchItem = current.item;
        if (!batchItem.smooth) {
          notifyProducerForSpace();
          current = undefined;
          yield batchItem.emit({
            text: batchItem.text,
            isFirst: true,
            isLast: true,
          });
          continue;
        }
        if (batchItem.text === '') {
          bufferedTextLength = Math.max(
            0,
            bufferedTextLength - current.textLength
          );
          notifyProducerForSpace();
          current = undefined;
          continue;
        }
        if (batchItem.atomic === true) {
          break;
        }

        const remainingText = batchItem.text.slice(headOffset);
        const pieceLength = findStreamChunkBoundary(
          remainingText,
          tickBudget - consumed
        );
        const pieceEnd = headOffset + pieceLength;
        const piece = batchItem.text.slice(headOffset, pieceEnd);
        const isFirst = headOffset === 0;
        const isLast = pieceEnd === batchItem.text.length;

        bufferedTextLength = Math.max(0, bufferedTextLength - piece.length);
        notifyProducerForSpace();
        if (isLast) {
          current = undefined;
        } else {
          headOffset = pieceEnd;
        }
        consumed += piece.length;
        yield batchItem.emit({ text: piece, isFirst, isLast });
      }
    }
  } finally {
    consumerClosed = true;
    if (producerState.done) {
      await producer;
    } else {
      abortUpstream?.();
      notifyProducerForSpace();
      const iterator = source as Partial<AsyncGenerator<SmoothItem<TEmit>>>;
      const closing = iterator.return?.call(source, undefined as never);
      if (closing != null) {
        void closing.then(
          () => undefined,
          () => undefined
        );
      }
      await Promise.race([
        producer,
        new Promise<void>((resolve) => {
          const timeout = setTimeout(resolve, PRODUCER_CLOSE_GRACE_MS);
          timeout.unref();
        }),
      ]);
    }
  }
}

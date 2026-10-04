import type { SmoothItem, SmoothPiece } from './smoother';
import {
  smoothStream,
  resolveStreamDelay,
  STREAM_ABORT_MESSAGE,
  STREAM_CHUNK_MIN_SIZE,
  DEFAULT_STREAM_DELAY,
  PRODUCER_CLOSE_GRACE_MS,
  MAX_SMOOTH_ITEM_SEGMENT_CHARS,
  STREAM_BOUNDARY_LOOKAHEAD_CHARS,
  findStreamChunkBoundary,
  computeAdaptivePieceSize,
  SMOOTH_TARGET_LATENCY_MS,
  MAX_STREAM_QUEUE_TEXT_CHARS,
} from './smoother';
import { ProviderTextAttempt } from '@/protection/providerText';

type Emitted = SmoothPiece & { tag: string; at: number };

function makeItem(
  tag: string,
  text: string,
  smooth: boolean,
  atomic?: boolean
): SmoothItem<Emitted> {
  return {
    text,
    smooth,
    atomic,
    emit: (piece) => ({ ...piece, tag, at: Date.now() }),
  };
}

async function* arraySource<T>(
  items: SmoothItem<T>[],
  onYield?: (index: number) => void
): AsyncGenerator<SmoothItem<T>> {
  for (let i = 0; i < items.length; i++) {
    onYield?.(i);
    yield items[i];
  }
}

async function collect(
  stream: AsyncGenerator<Emitted>
): Promise<Emitted[]> {
  const out: Emitted[] = [];
  for await (const piece of stream) {
    out.push(piece);
  }
  return out;
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

describe('resolveStreamDelay', () => {
  it('defaults to DEFAULT_STREAM_DELAY when unset', () => {
    expect(resolveStreamDelay(undefined)).toBe(DEFAULT_STREAM_DELAY);
  });

  it('passes explicit values through and clamps negatives to 0', () => {
    expect(resolveStreamDelay(10)).toBe(10);
    expect(resolveStreamDelay(0)).toBe(0);
    expect(resolveStreamDelay(-5)).toBe(0);
  });

  it('normalizes non-finite values to the default', () => {
    expect(resolveStreamDelay(Number.NaN)).toBe(DEFAULT_STREAM_DELAY);
    expect(resolveStreamDelay(Number.POSITIVE_INFINITY)).toBe(
      DEFAULT_STREAM_DELAY
    );
  });
});

describe('computeAdaptivePieceSize', () => {
  it('floors at STREAM_CHUNK_MIN_SIZE for small backlogs', () => {
    expect(computeAdaptivePieceSize(0, 25)).toBe(STREAM_CHUNK_MIN_SIZE);
    expect(computeAdaptivePieceSize(1, 25)).toBe(STREAM_CHUNK_MIN_SIZE);
    expect(computeAdaptivePieceSize(39, 25)).toBe(STREAM_CHUNK_MIN_SIZE);
  });

  it('scales proportionally with backlog', () => {
    expect(computeAdaptivePieceSize(250, 25)).toBe(25);
    expect(computeAdaptivePieceSize(2500, 25)).toBe(250);
    expect(computeAdaptivePieceSize(1000, 10)).toBe(40);
  });

  it('returns the full backlog when the tick is non-positive', () => {
    expect(computeAdaptivePieceSize(500, 0)).toBe(500);
  });

  it('holds steady-state backlog near arrival_rate x target latency', () => {
    const tick = 25;
    const arrivalPerTick = 100;
    let backlog = 0;
    const observed: number[] = [];
    for (let i = 0; i < 200; i++) {
      backlog += arrivalPerTick;
      backlog -= Math.min(backlog, computeAdaptivePieceSize(backlog, tick));
      observed.push(backlog);
    }
    const expected = (arrivalPerTick * SMOOTH_TARGET_LATENCY_MS) / tick;
    const tail = observed.slice(-20);
    for (const value of tail) {
      expect(value).toBeGreaterThan(expected * 0.8);
      expect(value).toBeLessThan(expected * 1.2);
    }
  });

  it('drains a stopped stream geometrically', () => {
    const tick = 25;
    const perTickTicks = Math.ceil(SMOOTH_TARGET_LATENCY_MS / tick);
    let backlog = 1000;
    let ticks = 0;
    const snapshots: number[] = [];
    while (backlog > 0 && ticks < 100) {
      backlog -= Math.min(backlog, computeAdaptivePieceSize(backlog, tick));
      ticks++;
      snapshots.push(backlog);
    }
    expect(backlog).toBe(0);
    expect(snapshots[perTickTicks - 1]).toBeLessThan(1000 * 0.4);
    expect(ticks).toBeLessThan(50);
  });
});

describe('findStreamChunkBoundary', () => {
  it('extends to the next boundary character inclusive', () => {
    expect(findStreamChunkBoundary('hello world', 4)).toBe(6);
  });

  it('returns full length when no boundary follows within the lookahead', () => {
    expect(findStreamChunkBoundary('hello', 4)).toBe(5);
    expect(findStreamChunkBoundary('hi', 4)).toBe(2);
  });

  it('hard-cuts boundary-free runs at the lookahead limit', () => {
    const unbroken = 'a'.repeat(10000);
    expect(findStreamChunkBoundary(unbroken, 4096)).toBe(
      4096 + STREAM_BOUNDARY_LOOKAHEAD_CHARS
    );
  });
});

describe('smoothStream', () => {
  it('splits a large smooth item into word-boundary pieces at cadence', async () => {
    const text =
      'The quick brown fox jumps over the lazy dog and keeps on running through the field. ';
    const pieces = await collect(
      smoothStream({
        source: arraySource([makeItem('a', text, true)]),
        delayMs: 5,
      })
    );

    expect(pieces.length).toBeGreaterThan(1);
    expect(pieces.map((p) => p.text).join('')).toBe(text);
    expect(pieces[0].isFirst).toBe(true);
    expect(pieces[pieces.length - 1].isLast).toBe(true);
    for (const piece of pieces.slice(1, -1)) {
      expect(piece.isFirst).toBe(false);
      expect(piece.isLast).toBe(false);
    }
  });

  it('preserves strict FIFO order across smooth and passthrough items', async () => {
    const items = [
      makeItem('text1', 'hello world again and again ', true),
      makeItem('tool', '', false),
      makeItem('text2', 'more visible text here now ', true),
      makeItem('usage', '', false),
    ];
    const pieces = await collect(
      smoothStream({ source: arraySource(items), delayMs: 3 })
    );

    const tagRuns = pieces.map((p) => p.tag);
    const collapsed = tagRuns.filter((tag, i) => tagRuns[i - 1] !== tag);
    expect(collapsed).toEqual(['text1', 'tool', 'text2', 'usage']);
  });

  it('emits the first visible piece without delay (TTFT)', async () => {
    const start = Date.now();
    const stream = smoothStream({
      source: arraySource([makeItem('a', 'hello world and more text ', true)]),
      delayMs: 50,
    });
    const first = await stream.next();
    expect(first.done).toBe(false);
    expect(Date.now() - start).toBeLessThan(40);
    await stream.return(undefined);
  });

  it('never delays passthrough items even between paced text', async () => {
    const items = [
      makeItem('text1', 'some visible words streaming along nicely ', true),
      makeItem('seal', '', false),
    ];
    const pieces = await collect(
      smoothStream({ source: arraySource(items), delayMs: 20 })
    );
    const sealPiece = pieces.find((p) => p.tag === 'seal');
    const lastText = pieces.filter((p) => p.tag === 'text1').pop();
    expect(sealPiece).toBeDefined();
    expect(lastText).toBeDefined();
    expect((sealPiece as Emitted).at - (lastText as Emitted).at).toBeLessThan(
      15
    );
  });

  it('paces atomic items whole without splitting', async () => {
    const text = 'this text carries logprobs and must never be split apart ';
    const pieces = await collect(
      smoothStream({
        source: arraySource([
          makeItem('lead', 'lead text goes first here ', true),
          makeItem('logprobs', text, true, true),
        ]),
        delayMs: 3,
      })
    );
    const atomicPieces = pieces.filter((p) => p.tag === 'logprobs');
    expect(atomicPieces).toHaveLength(1);
    expect(atomicPieces[0].text).toBe(text);
    expect(atomicPieces[0].isFirst).toBe(true);
    expect(atomicPieces[0].isLast).toBe(true);
  });

  it('marks exactly one isFirst piece per item for usage dedup', async () => {
    const pieces = await collect(
      smoothStream({
        source: arraySource([
          makeItem('a', 'first item with several words to split ', true),
          makeItem('b', 'second item also has words to split ', true),
        ]),
        delayMs: 2,
      })
    );
    for (const tag of ['a', 'b']) {
      const tagged = pieces.filter((p) => p.tag === tag);
      expect(tagged.filter((p) => p.isFirst)).toHaveLength(1);
      expect(tagged[0].isFirst).toBe(true);
    }
  });

  it('passes everything through unsplit and undelayed when delayMs is 0', async () => {
    const text = 'a very long text that would normally be split into pieces ';
    const start = Date.now();
    const pieces = await collect(
      smoothStream({
        source: arraySource([
          makeItem('a', text, true),
          makeItem('b', '', false),
        ]),
        delayMs: 0,
      })
    );
    expect(pieces).toHaveLength(2);
    expect(pieces[0].text).toBe(text);
    expect(pieces[0].isFirst).toBe(true);
    expect(pieces[0].isLast).toBe(true);
    expect(Date.now() - start).toBeLessThan(50);
  });

  it('coalesces token-sized items so backlog drains within the target window', async () => {
    const tick = 10;
    const items = Array.from({ length: 400 }, (_, i) =>
      makeItem(`t${i}`, i % 5 === 4 ? 'x ' : 'x', true)
    );
    const expected = items.map((i) => i.text).join('');

    const start = Date.now();
    const pieces = await collect(
      smoothStream({ source: arraySource(items), delayMs: tick })
    );
    const elapsed = Date.now() - start;

    expect(pieces.map((p) => p.text).join('')).toBe(expected);
    expect(elapsed).toBeLessThan(2000);
    const tags = pieces.map((p) => p.tag);
    expect(tags).toEqual([...tags].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1))));
  });

  it('rethrows a producer failure even when the rejection value is nullish', async () => {
    async function* failingSource(): AsyncGenerator<SmoothItem<Emitted>> {
      yield makeItem('a', 'some text ', true);
      throw undefined;
    }

    await expect(
      collect(smoothStream({ source: failingSource(), delayMs: 1 }))
    ).rejects.toThrow('Stream producer failed.');
  });

  it('stays fully lazy when smoothing is disabled', async () => {
    const yielded: number[] = [];
    const items = [
      makeItem('a', 'first ', true),
      makeItem('b', 'second ', true),
      makeItem('c', 'third ', true),
    ];
    const stream = smoothStream({
      source: arraySource(items, (i) => yielded.push(i)),
      delayMs: 0,
    });

    await stream.next();
    expect(yielded).toEqual([0]);
    await stream.return(undefined);
    expect(yielded).toEqual([0]);
  });

  it('segments an oversized item so pieces reassemble with exactly one isFirst/isLast', async () => {
    const bigText = 'lorem ipsum dolor sit amet '.repeat(400);
    const pieces = await collect(
      smoothStream({
        source: arraySource([makeItem('big', bigText, true)]),
        delayMs: 1,
      })
    );

    expect(pieces.map((p) => p.text).join('')).toBe(bigText);
    expect(pieces.filter((p) => p.isFirst)).toHaveLength(1);
    expect(pieces.filter((p) => p.isLast)).toHaveLength(1);
    expect(pieces[0].isFirst).toBe(true);
    expect(pieces[pieces.length - 1].isLast).toBe(true);
  });

  it('bounds pieces of boundary-free runs at the segment cap plus lookahead', async () => {
    const unbroken = 'x'.repeat(20000);
    const pieces = await collect(
      smoothStream({
        source: arraySource([makeItem('blob', unbroken, true)]),
        delayMs: 1,
      })
    );

    expect(pieces.map((p) => p.text).join('')).toBe(unbroken);
    const maxPiece = Math.max(...pieces.map((p) => p.text.length));
    expect(maxPiece).toBeLessThanOrEqual(
      MAX_SMOOTH_ITEM_SEGMENT_CHARS + STREAM_BOUNDARY_LOOKAHEAD_CHARS
    );
  });

  it('skips empty-text smooth items entirely', async () => {
    const pieces = await collect(
      smoothStream({
        source: arraySource([
          makeItem('empty', '', true),
          makeItem('real', 'actual text ', true),
        ]),
        delayMs: 3,
      })
    );
    expect(pieces.every((p) => p.tag === 'real')).toBe(true);
    expect(pieces.length).toBeGreaterThan(0);
  });

  it('wakes a consumer parked on an empty queue when the signal aborts', async () => {
    const controller = new AbortController();
    const abortUpstream = jest.fn();
    async function* stalledSource(): AsyncGenerator<SmoothItem<Emitted>> {
      yield makeItem('a', 'first text ', true);
      await new Promise<void>(() => undefined);
    }

    const stream = smoothStream({
      source: stalledSource(),
      delayMs: 5,
      signal: controller.signal,
      abortUpstream,
    });

    let next = await stream.next();
    while (next.done !== true && (next.value as Emitted).isLast !== true) {
      next = await stream.next();
    }

    const start = Date.now();
    const parked = stream.next();
    setTimeout(() => controller.abort(), 20);

    await expect(parked).rejects.toThrow(STREAM_ABORT_MESSAGE);
    expect(Date.now() - start).toBeLessThan(PRODUCER_CLOSE_GRACE_MS + 1500);
    expect(abortUpstream).toHaveBeenCalled();
  });

  it('completes an early break promptly even when the source is stalled', async () => {
    async function* stalledSource(): AsyncGenerator<SmoothItem<Emitted>> {
      yield makeItem('a', 'first text ', true);
      await new Promise<void>(() => undefined);
    }

    const stream = smoothStream({ source: stalledSource(), delayMs: 5 });
    await stream.next();

    const start = Date.now();
    const closed = (async (): Promise<boolean> => {
      await stream.return(undefined);
      return true;
    })();
    const result = await Promise.race([
      closed,
      sleep(PRODUCER_CLOSE_GRACE_MS + 2000).then(() => false),
    ]);
    expect(result).toBe(true);
    expect(Date.now() - start).toBeLessThan(PRODUCER_CLOSE_GRACE_MS + 1500);
  });

  it('throws the canonical abort message when aborted during the pacing sleep', async () => {
    const controller = new AbortController();
    const abortUpstream = jest.fn();
    const stream = smoothStream({
      source: arraySource([
        makeItem('a', 'plenty of text to keep the consumer pacing along ', true),
      ]),
      delayMs: 50,
      signal: controller.signal,
      abortUpstream,
    });

    const first = await stream.next();
    expect(first.done).toBe(false);
    setTimeout(() => controller.abort(), 10);

    await expect(async () => {
      await collect(stream as AsyncGenerator<Emitted>);
    }).rejects.toThrow(STREAM_ABORT_MESSAGE);
    expect(abortUpstream).toHaveBeenCalled();
  });

  it('parks the producer at the text capacity and resumes during consumer pacing', async () => {
    const bigText = 'word '.repeat(
      Math.ceil(MAX_STREAM_QUEUE_TEXT_CHARS / 5) + 200
    );
    const yielded: number[] = [];
    const items = [
      makeItem('a', bigText, true),
      makeItem('b', bigText, true),
      makeItem('c', 'tail text ', true),
    ];

    const stream = smoothStream({
      source: arraySource(items, (i) => yielded.push(i)),
      delayMs: 1,
    });

    const first = await stream.next();
    expect(first.done).toBe(false);
    /** Segmented admission parks the producer mid-way through the first
     * oversized item — later items are not read ahead past the text cap. */
    expect(yielded).toContain(0);
    expect(yielded).not.toContain(2);

    const rest = await collect(stream as AsyncGenerator<Emitted>);
    expect(yielded).toContain(1);
    expect(yielded).toContain(2);
    const all = [first.value as Emitted, ...rest];
    expect(all.map((p) => p.text).join('')).toBe(bigText + bigText + 'tail text ');
  });

  it('closes a parked producer promptly after an early consumer break', async () => {
    const bigText = 'word '.repeat(MAX_STREAM_QUEUE_TEXT_CHARS);
    let sourceFinallyRan = false;
    async function* source(): AsyncGenerator<SmoothItem<Emitted>> {
      try {
        yield makeItem('a', bigText, true);
        yield makeItem('b', bigText, true);
        yield makeItem('c', bigText, true);
      } finally {
        sourceFinallyRan = true;
      }
    }

    const abortUpstream = jest.fn();
    const stream = smoothStream({
      source: source(),
      delayMs: 25,
      abortUpstream,
    });

    await stream.next();
    await stream.next();

    const closed = (async (): Promise<boolean> => {
      await stream.return(undefined);
      return true;
    })();
    const result = await Promise.race([
      closed,
      sleep(1000).then(() => false),
    ]);
    expect(result).toBe(true);
    expect(abortUpstream).toHaveBeenCalled();
    expect(sourceFinallyRan).toBe(true);
  });

  it('keeps render lag bounded near the target latency under a fast big-chunk producer', async () => {
    const tick = 5;
    const chunk = 'lorem ipsum dolor sit amet consectetur adipiscing elit '.repeat(2);
    let producerDoneAt = 0;
    async function* fastSource(): AsyncGenerator<SmoothItem<Emitted>> {
      for (let i = 0; i < 12; i++) {
        yield makeItem(`c${i}`, chunk, true);
        await sleep(10);
      }
      producerDoneAt = Date.now();
    }

    const pieces = await collect(
      smoothStream({ source: fastSource(), delayMs: tick })
    );
    const renderDoneAt = pieces[pieces.length - 1].at;
    const lag = renderDoneAt - producerDoneAt;

    expect(pieces.map((p) => p.text).join('')).toBe(chunk.repeat(12));
    expect(lag).toBeLessThan(SMOOTH_TARGET_LATENCY_MS * 2.5);
  });
});

it('keeps protected positive-delay streams fully lazy without read-ahead or splitting', async () => {
  const attempt = new ProviderTextAttempt({ version: 1, timeoutMs: 10000, maxAttemptBytes: 1024, maxBufferedBytes: 1024, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
  const yielded: number[] = [];
  const source = arraySource([makeItem('first', 'Some allowed control text.', true), makeItem('second', 'Later control', true)], (i) => yielded.push(i));
  const stream = smoothStream({ source, delayMs: 25, signal: attempt.signal });
  try {
    const first = await stream.next();
    await new Promise((resolve) => setImmediate(resolve));
    expect(yielded).toEqual([0]);
    expect(first.value).toMatchObject({ text: 'Some allowed control text.', isFirst: true, isLast: true });
  } finally { await stream.return(undefined); attempt.finish(); }
});

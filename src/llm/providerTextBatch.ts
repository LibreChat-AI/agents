import { Buffer } from 'node:buffer';
import { performance } from 'node:perf_hooks';
import { AIMessageChunk } from '@langchain/core/messages';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type {
  ProviderTextAttempt,
  ProviderTextBatching,
} from '@/protection/providerText';
import { ProviderTextProtectionError } from '@/protection/providerText';

/** A single provider read can overlap inspection or downstream publication. */
export class ProviderTextBatch {
  private parts: string[] = [];
  private freshBytes = 0;
  private bytes = 0;
  private stagedText = 0;
  private stagedControl = 0;
  private reading = false;
  private next?: IteratorResult<ChatGenerationChunk>;
  private timer?: ReturnType<typeof setTimeout>;
  private deadline = Infinity;
  private wake?: () => void;
  private closed = false;

  constructor(
    private readonly attempt: ProviderTextAttempt,
    private readonly options: ProviderTextBatching,
    private readonly admit: (chunk: ChatGenerationChunk) => number
  ) {}

  private prefetch(source: AsyncGenerator<ChatGenerationChunk>): void {
    if (this.reading || this.next != null) return;
    this.reading = true;
    void this.attempt
      .wait(source.next())
      .then((next) => {
        this.reading = false;
        if (this.closed) return;
        this.attempt.check();
        if (next.done !== true) {
          const { content } = next.value.message;
          if (
            typeof content !== 'string' ||
            !(next.value.message instanceof AIMessageChunk)
          ) {
            throw new ProviderTextProtectionError('unsupported');
          }
          this.stagedControl = this.admit(next.value);
          // Raw fragments, their joined copy, and detached prefix/tail during a split.
          this.stagedText = content === '' ? 0 : content.length * 8 + 256;
          this.attempt.retain(this.stagedText + this.stagedControl);
        }
        this.next = next;
        this.wake?.();
      })
      .catch((error: unknown) => {
        if (this.closed) return;
        this.attempt.stop(
          error instanceof ProviderTextProtectionError
            ? error
            : new ProviderTextProtectionError('unavailable')
        );
        this.wake?.();
      });
  }

  async read(
    source: AsyncGenerator<ChatGenerationChunk>
  ): Promise<IteratorResult<ChatGenerationChunk> | 'flush'> {
    this.attempt.check();
    this.prefetch(source);
    for (;;) {
      this.attempt.check();
      if (
        this.freshBytes >= this.options.batchBytes ||
        performance.now() >= this.deadline
      )
        return 'flush';
      if (this.next != null) {
        const next = this.next;
        this.next = undefined;
        return next;
      }
      // Only one waiter, so a timer shared across a batch does not accumulate race listeners.
      const signal = this.attempt.signal;
      const ready = Promise.withResolvers<void>();
      this.wake = (): void => ready.resolve();
      signal.addEventListener('abort', this.wake, { once: true });
      try {
        if (!signal.aborted) await ready.promise;
      } finally {
        signal.removeEventListener('abort', this.wake);
        this.wake = undefined;
      }
    }
  }

  append(text: string): void {
    const bytes = this.stagedText;
    this.stagedText = 0;
    this.attempt.releaseWindow(this.stagedControl);
    this.stagedControl = 0;
    if (text === '') return;
    this.parts.push(text);
    this.freshBytes += text.length * 2;
    this.bytes += bytes;
    if (this.timer == null) {
      this.deadline = performance.now() + this.options.flushMs;
      this.timer = setTimeout(() => {
        this.deadline = 0;
        this.wake?.();
      }, this.options.flushMs);
      this.timer.unref();
    }
  }

  async flush(final: boolean): Promise<string> {
    this.resetTimer();
    this.freshBytes = 0;
    const text = this.parts.length === 1 ? this.parts[0] : this.parts.join('');
    if (text === '') return '';
    const approved = await this.attempt.inspect(text);
    if (approved !== text)
      throw new ProviderTextProtectionError('incompatible');
    if (final) return text;
    let end = Math.max(0, text.length - (this.options.maxMatchChars - 1));
    if (end > 0) {
      const previous = text.charCodeAt(end - 1);
      if (previous >= 0xd800 && previous <= 0xdbff) end--;
    }
    if (end === 0) return '';
    const tail = end === text.length ? '' : detach(text.slice(end));
    const prefix = tail === '' ? text : detach(text.slice(0, end));
    const retained = tail === '' ? 0 : tail.length * 8 + 256;
    this.parts = tail === '' ? [] : [tail];
    this.attempt.releaseWindow(this.bytes - retained);
    this.bytes = retained;
    return prefix;
  }

  private resetTimer(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.deadline = Infinity;
  }

  close(): void {
    this.closed = true;
    this.resetTimer();
    this.parts = [];
    this.next = undefined;
  }
}

/** Avoid keeping an unchecked suffix alive through a sliced, published prefix. */
function detach(text: string): string {
  return Buffer.from(text, 'utf16le').toString('utf16le');
}

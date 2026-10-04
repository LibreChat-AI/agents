// src/specs/preemptRestart.test.ts
/**
 * End-to-end discard-and-restart flow through a real `Run`.
 *
 * The companion to `preemptSeal.test.ts`: that file covers a preempt request
 * that arrives once the turn has an answer worth keeping. This one covers the
 * window BEFORE that — the model is silent, or streaming reasoning it has not
 * turned into text yet — where a seal is impossible and the request used to
 * wait out the whole turn. Here the in-flight call is torn down, its partial
 * output dropped, and the model called again with the boundary's injection
 * appended to the SAME prompt.
 *
 * Everything runs the dispatch-synchronous loop in `attemptInvoke` (no
 * registered CHAT_MODEL_STREAM handler), which is the only loop allowed to
 * seal or restart.
 */
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type { HookCallback } from '@/hooks/types';
import type * as t from '@/types';
import { HookRegistry } from '@/hooks/HookRegistry';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

const RESTARTED_RESPONSE = 'Answering the steer instead.';

/**
 * A reasoning-only chunk. The seal gate rejects these by design — several
 * providers strip or sign reasoning, so it cannot stand in for the visible
 * assistant turn a sealed sequence needs — which is exactly why a turn made of
 * nothing else is discardable rather than sealable.
 */
function thinkingChunk(text: string): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: [{ type: 'thinking', thinking: text }],
    }),
  });
}

const streamConfig = {
  configurable: { thread_id: 'preempt-restart-e2e' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

/**
 * A host arming an interrupt, modelled the way a real one behaves: the flag is
 * level-triggered and stays true until a boundary drains it, and `wake` is a
 * hint the SDK may act on or ignore.
 */
function createHost(
  restartGraceMs: number,
  /**
   * When false the channel is registered but never fired. That is not a broken
   * host — a wake that arrives while the shape is not yet discardable declines
   * and never comes again — and it is the only deterministic way to reach the
   * per-chunk trigger, which ends the stream by BREAKING the iterator rather
   * than aborting it.
   */
  deliverWakes = true
): {
  arm: () => void;
  disarm: () => void;
  preemption: t.StreamPreemption;
} {
  let armed = false;
  const wakes = new Set<() => void>();
  return {
    arm: (): void => {
      armed = true;
      if (!deliverWakes) {
        return;
      }
      for (const wake of wakes) {
        wake();
      }
    },
    disarm: (): void => {
      armed = false;
    },
    preemption: {
      shouldPreempt: () => armed,
      subscribe: (wake) => {
        wakes.add(wake);
        return () => {
          wakes.delete(wake);
        };
      },
      restartGraceMs,
      maxSeals: 2,
    },
  };
}

/** Records every prompt the provider was actually asked to complete. */
class PromptRecordingModel extends FakeChatModel {
  prompts: BaseMessage[][] = [];
  /**
   * Fired from inside the first stream, so a test can arm the interrupt at a
   * realistic moment: while the provider is already working. Arming before
   * `processStream` instead exercises a different path entirely — the request
   * is outstanding before the call is issued, and the call never goes out.
   */
  onFirstStream: () => void = () => {};

  protected recordPrompt(messages: BaseMessage[]): number {
    this.prompts.push(messages);
    return this.prompts.length;
  }
}

/**
 * First call hangs without yielding anything — the silent window between a
 * request and its first chunk, where no per-chunk poll can ever run. Later
 * calls answer normally.
 */
class SilentThenAnsweringModel extends PromptRecordingModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) === 1) {
      this.onFirstStream();
      await new Promise<never>((_resolve, reject) => {
        const { signal } = options;
        if (signal == null) {
          reject(new Error('Expected the attempt to carry an abort signal'));
          return;
        }
        signal.addEventListener(
          'abort',
          () => reject(new Error('provider stream aborted')),
          { once: true }
        );
      });
      return;
    }
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

/**
 * First call streams reasoning and nothing else, then hangs — the long
 * thinking stretch. `thinkingChunkCount` is deliberately more than one so the
 * per-chunk trigger is exercised repeatedly against the grace window.
 */
class ThinkingThenAnsweringModel extends PromptRecordingModel {
  thinkingChunkCount = 3;

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) !== 1) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }
    for (let i = 0; i < this.thinkingChunkCount; i++) {
      yield thinkingChunk(`step ${i}`);
      this.onFirstStream();
    }
    await new Promise<never>((_resolve, reject) => {
      options.signal?.addEventListener(
        'abort',
        () => reject(new Error('provider stream aborted')),
        { once: true }
      );
    });
  }
}

/**
 * Ignores cancellation entirely — `stream` is overridden so LangChain's own
 * signal handling never runs, leaving the SDK's consumer loop as the only
 * thing that can stop it. That is the adapter the abort cannot be trusted
 * against, and the buffered-chunk case behaves the same way from the loop's
 * side. `yielded` records how far it got before the consumer closed it.
 */
class UncooperativeModel extends PromptRecordingModel {
  yielded = 0;
  readonly textChunks = 40;

  override stream(
    input: Parameters<FakeChatModel['stream']>[0],
    options?: Parameters<FakeChatModel['stream']>[1]
  ): ReturnType<FakeChatModel['stream']> {
    const messages = Array.isArray(input) ? (input as BaseMessage[]) : [];
    if (this.recordPrompt(messages) !== 1) {
      return super.stream(input, options);
    }
    const countYield = (): void => {
      this.yielded += 1;
    };
    const arm = (): void => this.onFirstStream();
    const { textChunks } = this;
    async function* uncooperative(): AsyncGenerator<AIMessageChunk> {
      yield new AIMessageChunk({
        content: [{ type: 'thinking', thinking: 'deciding' }],
      });
      countYield();
      arm();
      for (let i = 0; i < textChunks; i++) {
        yield new AIMessageChunk({ content: 'x' });
        countYield();
      }
    }
    return uncooperative() as unknown as ReturnType<FakeChatModel['stream']>;
  }
}

/** Streams reasoning and then ends, never producing text. */
class ThinkingOnlyModel extends PromptRecordingModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) !== 1) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }
    yield thinkingChunk('first');
    this.onFirstStream();
    yield thinkingChunk('second');
  }
}

/** Streams reasoning first, then real text — the turn a seal must win. */
class ThinkingThenTextModel extends PromptRecordingModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) !== 1) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }
    yield thinkingChunk('almost there');
    this.onFirstStream();
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

/** A host on the seal-only contract: armed, but with no wake channel. */
function createSealOnlyHost(): {
  arm: () => void;
  disarm: () => void;
  preemption: t.StreamPreemption;
  } {
  let armed = false;
  return {
    arm: (): void => {
      armed = true;
    },
    disarm: (): void => {
      armed = false;
    },
    preemption: { shouldPreempt: () => armed, maxSeals: 2 },
  };
}

async function createRestartRun(options: {
  runId: string;
  hook: HookCallback<'PreemptBoundary'>;
  preemption: t.StreamPreemption;
  model: FakeChatModel;
  protectedText?: boolean;
}): Promise<Run<t.IState>> {
  const registry = new HookRegistry();
  registry.register('PreemptBoundary', { hooks: [options.hook] });
  const run = await Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4o-mini',
        apiKey: 'test-key',
      },
      instructions: 'Answer plainly.',
    },
    hooks: registry,
    ...(options.protectedText === true ? { providerTextProtection: {
      version: 1 as const, timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144,
      classify: (): 'prose' => 'prose',
      inspect: ({ content }: { content: string }) => ({ version: 1 as const, ok: true as const, value: { content, replacements: 0, categories: [] } }),
    } } : {}),
    preemption: options.preemption,
    /**
     * Required for a discarded turn to report usage at all: a turn that ends
     * early never receives the provider's usage chunk, so the synthetic close
     * falls back to this counter. Without one the whole usage path silently
     * no-ops and any test watching it proves nothing.
     */
    tokenCounter: (message: BaseMessage): number =>
      Math.ceil(JSON.stringify(message.content).length / 4),
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  run.Graph.overrideModel = options.model;
  return run;
}

function userTexts(messages: BaseMessage[]): string[] {
  const texts: string[] = [];
  for (const message of messages) {
    if (message.getType() !== 'human') {
      continue;
    }
    texts.push(
      typeof message.content === 'string'
        ? message.content
        : JSON.stringify(message.content)
    );
  }
  return texts;
}

describe('cooperative restart (end-to-end via Run)', () => {
  it.each([false, true])('discards a silent turn and re-issues it with the steer appended (protected=%s)', async (protectedText) => {
    const host = createHost(0);
    /**
     * The screenshot case: the provider has accepted the request and gone
     * quiet. No chunk will ever arrive to poll on, so the wake is the only
     * thing that can reach this turn.
     */
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: `restart-silent-${protectedText}`,
      protectedText,
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [
            { role: 'user' as const, content: 'actually, be brief' },
          ],
        };
      },
    });
    model.onFirstStream = protectedText ? () => { setImmediate(host.arm); } : host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    /**
     * The whole safety argument in one assertion: the discarded turn leaves no
     * assistant message behind, so the injected turn sits directly after the
     * original question. Adjacent user turns are native on Anthropic, OpenAI
     * and Gemini, and normalized for the strict-alternation providers.
     */
    expect(userTexts(model.prompts[1])).toEqual([
      'tell me a long story',
      'actually, be brief',
    ]);
    expect(
      model.prompts[1].some((message) => message.getType() === 'ai')
    ).toBe(false);
  });

  it.each([false, true])('discards a thinking-only turn once the request outlives the grace (protected=%s)', async (protectedText) => {
    const host = createHost(0);
    const model = new ThinkingThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: `restart-thinking-${protectedText}`,
      protectedText,
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop there' }],
        };
      },
    });
    model.onFirstStream = protectedText ? () => { setImmediate(host.arm); } : host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    expect(userTexts(model.prompts[1])).toEqual([
      'think it through',
      'stop there',
    ]);
  });

  it('prefers a seal when text arrives inside the grace window', async () => {
    const host = createHost(60_000);
    const model = new ThinkingThenTextModel({
      responses: ['Partial answer.', RESTARTED_RESPONSE],
    });
    const boundaryPrompts: number[] = [];
    const run = await createRestartRun({
      runId: 'restart-defers-to-seal',
      preemption: host.preemption,
      model,
      hook: () => {
        boundaryPrompts.push(model.prompts.length);
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'go on' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think then answer')] },
      streamConfig
    );

    expect(boundaryPrompts).toHaveLength(1);
    expect(model.prompts).toHaveLength(2);
    /**
     * A seal KEEPS the turn, so the resumed prompt carries the partial
     * assistant answer. That is the whole difference from a restart, and the
     * grace window is what guarantees it: reasoning alone must not convert a
     * turn whose text was moments away.
     */
    expect(
      model.prompts[1].some((message) => message.getType() === 'ai')
    ).toBe(true);
  });

  it('re-issues the call when a restart boundary injects nothing', async () => {
    const host = createHost(0);
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-empty-boundary',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {};
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    /**
     * A cancelled interrupt must not cost the user their answer. A seal cannot
     * self-loop here (it would leave a trailing model turn with no new input),
     * but a discard left graph state untouched, so re-entering the node simply
     * re-issues the original call.
     */
    expect(model.prompts).toHaveLength(2);
    expect(userTexts(model.prompts[1])).toEqual(['tell me a long story']);
  });
});

/**
 * Everything a restart depends on is opt-in through `subscribe`, and the lane
 * it names is where the discard's boundary gets dispatched. Without it a
 * discard would return no message AND record no lane, so the node would inject
 * nothing and end the turn empty with the steer still queued.
 */
describe('seal-only hosts', () => {
  it('never discards a thinking turn when no wake channel was supplied', async () => {
    const host = createSealOnlyHost();
    const model = new ThinkingThenTextModel({
      responses: ['Partial answer.', RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'seal-only-no-restart',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'go on' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think then answer')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    /**
     * Sealed, not discarded: the resumed prompt carries the partial assistant
     * answer. An empty `messages[1]` with no assistant turn would mean the
     * turn was thrown away with no boundary to catch it.
     */
    expect(model.prompts[1].some((message) => message.getType() === 'ai')).toBe(
      true
    );
  });

  /**
   * The turn a seal can never accept, on a host that cannot restart. It must
   * end as an ordinary reasoning-only turn — one model call, no boundary — and
   * NOT be discarded into an empty turn with the steer still queued.
   */
  it('lets a thinking-only turn finish rather than discarding it', async () => {
    const host = createSealOnlyHost();
    const model = new ThinkingOnlyModel({ responses: [RESTARTED_RESPONSE] });
    const boundaries: number[] = [];
    const run = await createRestartRun({
      runId: 'seal-only-thinking-only',
      preemption: host.preemption,
      model,
      hook: () => {
        boundaries.push(model.prompts.length);
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop there' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(1);
    expect(boundaries).toHaveLength(0);
  });
});

/**
 * The abort is a request, not a guarantee. An adapter that ignores it — or one
 * whose iterator hands over a chunk buffered before it landed — would keep
 * feeding the host text it is about to be told was discarded, and in the
 * ignoring case would hold the boundary until the whole response finished.
 */
describe('an adapter that ignores the abort', () => {
  it('stops being consumed as soon as the restart is decided', async () => {
    const host = createHost(0);
    const model = new UncooperativeModel({ responses: [RESTARTED_RESPONSE] });
    const run = await createRestartRun({
      runId: 'restart-uncooperative',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('go')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    /** A couple of chunks may already be in flight; the whole response must
     *  not be. */
    expect(model.yielded).toBeLessThan(model.textChunks);
    expect(userTexts(model.prompts[1])).toEqual(['go', 'stop']);
    expect(model.prompts[1].some((message) => message.getType() === 'ai')).toBe(
      false
    );
  });
});

/**
 * A restart preserves nothing, so reporting it as a seal would tell every
 * consumer of these counters that a turn was kept when it was discarded.
 */
describe('preempt stats', () => {
  it('counts a discard apart from a seal', async () => {
    const host = createHost(0);
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-stats',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'be brief' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    const stats = run.getPreemptStats();
    expect(stats.restarts).toBe(1);
    expect(stats.seals).toBe(0);
    /** A restart boundary that DID inject is not an empty boundary either. */
    expect(stats.emptyBoundaries).toBe(0);
  });

  it('does not count a reissued empty boundary as truncated', async () => {
    const host = createHost(0);
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-stats-empty',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {};
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    expect(run.getPreemptStats().emptyBoundaries).toBe(0);
  });
});

/**
 * The grace guards against stealing a seal that was about to become possible.
 * A stream that ENDS while still discardable is proof none was coming, so the
 * request must not be left for the terminal boundary — which would replay a
 * reasoning-only turn followed by the steer, the very shape the seal gate
 * refuses to build.
 */
describe('a discardable stream that ends inside the grace', () => {
  it('still honors the armed request', async () => {
    const host = createHost(60_000);
    const model = new ThinkingOnlyModel({ responses: [RESTARTED_RESPONSE] });
    const run = await createRestartRun({
      runId: 'restart-natural-end',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop there' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    expect(userTexts(model.prompts[1])).toEqual([
      'think it through',
      'stop there',
    ]);
    expect(model.prompts[1].some((message) => message.getType() === 'ai')).toBe(
      false
    );
  });
});

/**
 * Each way a restart can end the stream leaves the model run in a different
 * state, and closing one the way another needs either double-closes a finished
 * run or leaves an open one behind.
 */
describe('restart close routes', () => {
  it('does not close a run the stream already finished', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation();
    try {
      const host = createHost(60_000);
      const model = new ThinkingOnlyModel({ responses: [RESTARTED_RESPONSE] });
      const run = await createRestartRun({
        runId: 'restart-no-double-close',
        preemption: host.preemption,
        model,
        hook: () => {
          host.disarm();
          return {
            injectedMessages: [{ role: 'user' as const, content: 'stop' }],
          };
        },
      });
      model.onFirstStream = host.arm;

      await run.processStream(
        { messages: [new HumanMessage('think')] },
        streamConfig
      );

      expect(model.prompts).toHaveLength(2);
      /**
       * The natural end already emitted its own close. A second attempt finds
       * no run to end and falls back with this warning, which is the
       * observable signature of getting the route wrong.
       */
      expect(
        warn.mock.calls.filter(([message]) =>
          String(message).includes('Native close of the sealed model run')
        )
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  /**
   * A stream that finished on its own already gave the host its model-end,
   * with usage. Replaying that event would charge the discarded attempt's
   * tokens twice in any host accumulating from it.
   */
  it('does not replay model-end usage after natural exhaustion', async () => {
    const modelEnds: Array<Record<string, unknown> | undefined> = [];
    const host = createHost(60_000);
    const model = new ThinkingOnlyModel({ responses: [RESTARTED_RESPONSE] });
    const run = await createRestartRun({
      runId: 'restart-no-duplicate-usage',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think')] },
      {
        ...streamConfig,
        callbacks: [
          {
            /**
             * The synthetic close is dispatched as a CUSTOM event, which is
             * how hosts accumulating usage receive it — `handleLLMEnd` never
             * sees it, so watching that channel would miss the duplicate
             * entirely.
             */
            handleCustomEvent: (
              event: string,
              data: { output?: AIMessageChunk }
            ): void => {
              if (event === GraphEvents.CHAT_MODEL_END) {
                modelEnds.push(
                  data.output?.usage_metadata as unknown as Record<
                    string,
                    unknown
                  >
                );
              }
            },
          },
        ],
      }
    );

    expect(model.prompts).toHaveLength(2);
    /**
     * The natural end already reported this turn through LangChain. A
     * synthetic close carrying usage on top of it is the duplicate charge.
     */
    expect(modelEnds.filter((usage) => usage != null)).toHaveLength(0);
  });

  it('marks a manually closed restart as control flow', async () => {
    const outputs: Array<Record<string, unknown> | undefined> = [];
    const host = createHost(0, false);
    const model = new ThinkingThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-control-flow-output',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      {
        ...streamConfig,
        callbacks: [
          {
            handleLLMEnd: (output: LLMResult): void => {
              outputs.push(output.llmOutput as Record<string, unknown>);
            },
          },
        ],
      }
    );

    /**
     * A discarded turn must not read as an ordinary generation just because
     * the per-chunk trigger fired instead of the wake.
     */
    expect(outputs).toContainEqual(
      expect.objectContaining({ controlFlow: 'PreemptRestart' })
    );
  });
});

/**
 * A halted restart truncates nothing that was kept, so it is not the
 * truncated-seal signal hosts persist on. `preemptIncomplete` already records
 * that the answer never arrived.
 */
describe('a restart boundary that halts', () => {
  it('is not counted as an empty boundary', async () => {
    const host = createHost(0);
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-halted',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return { preventContinuation: true };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    const stats = run.getPreemptStats();
    expect(stats.restarts).toBe(1);
    expect(stats.emptyBoundaries).toBe(0);
  });
});

/**
 * A restart must never swallow a genuine failure that landed at the same
 * moment. The catch exists for one thing — the abort this attempt issued —
 * and a stream-limit trip or a real cancellation has to keep travelling.
 */
describe('a failure concurrent with a restart', () => {
  it('propagates a run cancellation rather than restarting through it', async () => {
    const host = createHost(0);
    const controller = new AbortController();
    let boundaries = 0;
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-concurrent-cancel',
      preemption: host.preemption,
      model,
      hook: () => {
        boundaries += 1;
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop' }],
        };
      },
    });
    model.onFirstStream = (): void => {
      controller.abort();
      host.arm();
    };

    await expect(
      run.processStream({ messages: [new HumanMessage('go')] }, {
        ...streamConfig,
        signal: controller.signal,
      })
    ).rejects.toBeDefined();

    /**
     * The discriminator: swallowing the cancellation lets the node reach its
     * boundary and consume the queued injection on a run the caller already
     * cancelled. Nothing downstream of the model node may run at all.
     */
    expect(boundaries).toBe(0);
    expect(model.prompts).toHaveLength(1);
  });
});

/**
 * Graph state forgets a discarded attempt, but the run-step view is a separate
 * projection: left alone it closes the attempt `completed`, so subscribers keep
 * showing reasoning the replacement prompt does not contain.
 */
describe('run steps for a discarded attempt', () => {
  it('leaves no step open behind it', async () => {
    const host = createHost(0);
    const model = new ThinkingThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-run-steps',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      streamConfig
    );

    /**
     * The reachable invariant. A step the discard cut short must not stay
     * `in_progress` — subscribers would show the withdrawn attempt as still
     * running. Whether such a step exists at all depends on what the provider
     * streamed before the interrupt, so this asserts the property rather than
     * a particular step count.
     */
    const statuses = (run.Graph?.getRunSteps() ?? []).map(
      (step) => step.status
    );
    expect(statuses.length).toBeGreaterThan(0);
    expect(
      statuses.filter((status) => status == null || status === 'in_progress')
    ).toHaveLength(0);
  });
});

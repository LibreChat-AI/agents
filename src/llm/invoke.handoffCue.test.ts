// src/llm/invoke.handoffCue.test.ts
/**
 * The handoff cue is re-keyed on the provider ACTUALLY serving the call at
 * the `attemptInvoke` funnel — the seam every fallback send passes through.
 * A tolerant primary falling back to a Claude surface must gain the cue; an
 * Anthropic primary falling back to a tolerant provider must have the baked
 * cue stripped; and the serving model id must be read through the wrapper
 * stack, or a wrapped Bedrock-Nova model would default to Claude.
 */
import { RunnableBinding } from '@langchain/core/runnables';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type { InvokeContext } from './invoke';
import {
  PREDECESSOR_HANDOFF_CUE,
  INSTRUCTIONLESS_HANDOFF_CUE,
  withInstructionlessHandoffCue,
} from '@/messages/handoffCue';
import { FakeChatModel } from '@/llm/fake';
import { attemptInvoke } from './invoke';
import { Providers } from '@/common';

class CapturingChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];
  /** Serving model id, as a real provider client would expose it. */
  model?: string;

  constructor(modelId?: string) {
    super({ responses: ['ok'] });
    this.model = modelId;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

const runTail = new AIMessage({ content: 'predecessor output', id: 'run-1' });
const context = {
  isRunProducedMessage: (message: BaseMessage): boolean =>
    message.id === 'run-1',
  getOrCreateToolOutputRegistry: (): undefined => undefined,
} as unknown as InvokeContext;

const bakedCue = (): HumanMessage =>
  new HumanMessage({
    content: PREDECESSOR_HANDOFF_CUE,
    additional_kwargs: { role: 'user', isMeta: true, source: 'handoff' },
  });

async function sentBy(
  provider: Providers,
  options?: { modelId?: string; wrap?: boolean; messages?: BaseMessage[] }
): Promise<BaseMessage[]> {
  const model = new CapturingChatModel(options?.modelId);
  const served =
    options?.wrap === true
      ? new RunnableBinding({ bound: model, kwargs: {}, config: {} })
      : model;
  await attemptInvoke({
    model: served as never,
    messages: options?.messages ?? [new HumanMessage('go'), runTail],
    provider,
    context,
    onChunk: async () => undefined,
  });
  return model.invocations[0];
}

describe('attemptInvoke handoff-cue funnel', () => {
  it('grounds repeated gateway attempts without mutating their replay source', async () => {
    const model = new CapturingChatModel('analytics-gateway');
    const messages = [new HumanMessage('go'), runTail];
    const config = withInstructionlessHandoffCue(undefined, runTail);
    const stream = model._streamResponseChunks.bind(model);
    let failed = false;
    jest
      .spyOn(model, '_streamResponseChunks')
      .mockImplementation(async function* (payload, options, runManager) {
        if (!failed) {
          failed = true;
          model.invocations.push(payload);
          throw new Error('Transient gateway failure');
        }
        yield* stream(payload, options, runManager);
      });
    const invoke = (): ReturnType<typeof attemptInvoke> =>
      attemptInvoke(
        {
          model,
          messages,
          provider: Providers.OPENAI,
          onChunk: async () => undefined,
        },
        config
      );
    await expect(invoke()).rejects.toThrow('Transient gateway failure');
    await invoke();
    expect(model.invocations).toHaveLength(2);
    for (const request of model.invocations) {
      expect(request.at(-1)?.content).toBe(INSTRUCTIONLESS_HANDOFF_CUE);
      expect(
        request.filter(
          (message) => message.content === INSTRUCTIONLESS_HANDOFF_CUE
        )
      ).toHaveLength(1);
    }
    expect(messages).toEqual([new HumanMessage('go'), runTail]);
  });

  it('applies the cue when the serving provider is Anthropic', async () => {
    const sent = await sentBy(Providers.ANTHROPIC);
    expect(sent.at(-1)?.content).toBe(PREDECESSOR_HANDOFF_CUE);
  });

  it('applies the cue for Bedrock serving a WRAPPED Claude model', async () => {
    const sent = await sentBy(Providers.BEDROCK, {
      modelId: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
      wrap: true,
    });
    expect(sent.at(-1)?.content).toBe(PREDECESSOR_HANDOFF_CUE);
  });

  it('does not apply the cue for Bedrock serving a WRAPPED Nova model', async () => {
    const sent = await sentBy(Providers.BEDROCK, {
      modelId: 'us.amazon.nova-pro-v1:0',
      wrap: true,
    });
    expect(sent.at(-1)?.getType()).toBe('ai');
  });

  it('strips a baked cue for a tolerant serving provider', async () => {
    const sent = await sentBy(Providers.OPENAI, {
      messages: [new HumanMessage('go'), runTail, bakedCue()],
    });
    expect(sent.at(-1)?.getType()).toBe('ai');
    expect(sent.some((m) => m.content === PREDECESSOR_HANDOFF_CUE)).toBe(false);
  });

  it('is idempotent when the primary already baked the cue', async () => {
    const sent = await sentBy(Providers.ANTHROPIC, {
      messages: [new HumanMessage('go'), runTail, bakedCue()],
    });
    expect(
      sent.filter((m) => m.content === PREDECESSOR_HANDOFF_CUE)
    ).toHaveLength(1);
  });
});

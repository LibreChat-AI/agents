import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  appendPredecessorHandoffCue,
  removePredecessorHandoffCue,
  PREDECESSOR_HANDOFF_CUE,
  appendInstructionlessHandoffCue,
  withInstructionlessHandoffCue,
  INSTRUCTIONLESS_HANDOFF_CUE,
} from './handoffCue';
import { getProviderMessageProvenance } from './provenance';

const runAi = new AIMessage({ content: 'predecessor output', id: 'run-ai-1' });
const producedIds = new Set(['run-ai-1']);
const isRunProduced = (message: BaseMessage): boolean =>
  message.id != null && producedIds.has(message.id);

describe('instructionless handoff cue', () => {
  const config = withInstructionlessHandoffCue(undefined, runAi);

  it('is wire-only, synthetic, idempotent, and retained for tolerant fallback providers', () => {
    const messages = [runAi];
    const cued = appendInstructionlessHandoffCue(messages, config);
    expect(cued.at(-1)?.content).toBe(INSTRUCTIONLESS_HANDOFF_CUE);
    expect(getProviderMessageProvenance(cued.at(-1)!)?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
    expect(appendInstructionlessHandoffCue(cued, config)).toBe(cued);
    expect(removePredecessorHandoffCue(cued)).toBe(cued);
    expect(messages).toEqual([runAi]);
    expect(runAi.additional_kwargs).toEqual({});
  });

  it('matches provider clones but not later assistant turns or tool iterations', () => {
    const clone = new AIMessage({ content: 'projected copy', id: runAi.id });
    expect(appendInstructionlessHandoffCue([clone], config)).toHaveLength(2);
    const later = [
      runAi,
      new AIMessage({ content: 'next response', id: 'next' }),
    ];
    expect(appendInstructionlessHandoffCue(later, config)).toBe(later);
    const toolTail = [
      runAi,
      new ToolMessage({ content: 'result', tool_call_id: 'lookup' }),
    ];
    expect(appendInstructionlessHandoffCue(toolTail, config)).toBe(toolTail);
  });

  it('resets inherited scope without mutating the parent or a parallel sibling', () => {
    const reset = withInstructionlessHandoffCue(config);
    const other = new AIMessage({ content: 'other handoff', id: 'other' });
    const sibling = withInstructionlessHandoffCue(config, other);
    const messages = [runAi];
    expect(appendInstructionlessHandoffCue(messages, reset)).toBe(messages);
    expect(appendInstructionlessHandoffCue(messages, sibling)).toBe(messages);
    expect(appendInstructionlessHandoffCue(messages, config)).toHaveLength(2);
  });

  it('leaves deliberate prefill and user text alone outside a scoped handoff', () => {
    const prefill = [runAi];
    expect(appendInstructionlessHandoffCue(prefill)).toBe(prefill);
    const userText = [new HumanMessage(INSTRUCTIONLESS_HANDOFF_CUE)];
    expect(appendInstructionlessHandoffCue(userText, config)).toBe(userText);
    const idless = [new AIMessage('host prefill')];
    expect(
      appendInstructionlessHandoffCue(
        idless,
        withInstructionlessHandoffCue(undefined, idless[0])
      )
    ).toBe(idless);
  });
});

describe('appendPredecessorHandoffCue', () => {
  it('appends the cue when the payload ends with a run-produced assistant turn', () => {
    const payload = [new HumanMessage('ask'), runAi];
    const result = appendPredecessorHandoffCue(payload, isRunProduced);
    expect(result).toHaveLength(3);
    expect(result[2].content).toBe(PREDECESSOR_HANDOFF_CUE);
    expect(result[2].additional_kwargs).toMatchObject({
      role: 'user',
      isMeta: true,
      source: 'handoff',
    });
    expect(getProviderMessageProvenance(result[2])?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
    /** Non-mutating: the input array is untouched. */
    expect(payload).toHaveLength(2);
  });

  it('matches a transformed clone of the run tail by id', () => {
    const clone = new AIMessage({ content: 'projected copy', id: 'run-ai-1' });
    expect(appendPredecessorHandoffCue([clone], isRunProduced)).toHaveLength(2);
  });

  /**
   * Host-supplied trailing assistant turns are deliberate prefill (continue
   * generation flows). The run has not produced them, so provenance never
   * matches and the cue must stay off.
   */
  it('leaves host-history prefill payloads alone', () => {
    const hostAi = new AIMessage({ content: 'host prefill', id: 'host-ai-9' });
    expect(appendPredecessorHandoffCue([hostAi], isRunProduced)).toHaveLength(
      1
    );
    expect(appendPredecessorHandoffCue([hostAi], undefined)).toHaveLength(1);
  });

  it('is off when the payload does not end on an assistant turn', () => {
    const tool = new ToolMessage({ content: 'r', tool_call_id: 't1' });
    expect(
      appendPredecessorHandoffCue([runAi, tool], isRunProduced)
    ).toHaveLength(2);
    expect(
      appendPredecessorHandoffCue([new HumanMessage('hi')], isRunProduced)
    ).toHaveLength(1);
    expect(appendPredecessorHandoffCue([], isRunProduced)).toHaveLength(0);
  });

  it('fails safe when the tail carries no id', () => {
    const noId = new AIMessage({ content: 'no id' });
    expect(appendPredecessorHandoffCue([noId], isRunProduced)).toHaveLength(1);
  });
});

describe('removePredecessorHandoffCue', () => {
  const cue = new HumanMessage({
    content: PREDECESSOR_HANDOFF_CUE,
    additional_kwargs: { role: 'user', isMeta: true, source: 'handoff' },
  });

  it('strips a trailing cue and only a trailing cue', () => {
    const stripped = removePredecessorHandoffCue([runAi, cue]);
    expect(stripped).toHaveLength(1);
    expect(stripped[0]).toBe(runAi);
  });

  it('is identity when no cue trails', () => {
    const messages = [runAi];
    expect(removePredecessorHandoffCue(messages)).toBe(messages);
    const userTail = [runAi, new HumanMessage('real user turn')];
    expect(removePredecessorHandoffCue(userTail)).toBe(userTail);
    const empty: BaseMessage[] = [];
    expect(removePredecessorHandoffCue(empty)).toBe(empty);
  });

  it('does not strip a user turn that merely repeats the cue text', () => {
    const impostor = new HumanMessage({ content: PREDECESSOR_HANDOFF_CUE });
    const messages = [runAi, impostor];
    expect(removePredecessorHandoffCue(messages)).toBe(messages);
  });

  it('round-trips with append', () => {
    const payload = [new HumanMessage('ask'), runAi];
    const appended = appendPredecessorHandoffCue(payload, isRunProduced);
    expect(removePredecessorHandoffCue(appended)).toEqual(payload);
  });
});

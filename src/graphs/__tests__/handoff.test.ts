import { Command } from '@langchain/langgraph';
import { AIMessage } from '@langchain/core/messages';
import type { BaseGraphState, HandoffState } from '@/types';
import {
  HandoffRouting,
  HandoffLimitError,
  mergeHandoffState,
} from '../handoff';

function command(targetAgentId = 'B'): Command {
  return new Command({
    graph: Command.PARENT,
    goto: targetAgentId,
    update: {
      handoffRequest: {
        sourceAgentId: 'A',
        targetAgentId,
        toolCallId: 'transfer',
        scope: 'conversation',
      },
    },
  });
}
function state(handoffState: HandoffState): BaseGraphState {
  return {
    messages: [new AIMessage({ id: 'message', content: '' })],
    handoffState,
  };
}
const config = { configurable: { checkpoint_ns: 'stable-task' } };

describe('handoff admission and checkpoint reducer', () => {
  it('does not charge replay twice after reconstructing the owner', () => {
    const first = new HandoffRouting('A', 1, false);
    const input = state(first.start());
    first.finalize([command()], input, config);
    const rebuilt = new HandoffRouting('A', 1, false);
    rebuilt.resume(first.snapshot());
    rebuilt.finalize([command()], input, config);
    expect(rebuilt.snapshot().transitions).toEqual(
      first.snapshot().transitions
    );
    expect(rebuilt.snapshot().transitions).toHaveLength(1);
    expect(() =>
      rebuilt.finalize([command()], input, {
        configurable: { checkpoint_ns: 'new-task' },
      })
    ).toThrow(HandoffLimitError);
  });

  it('rejects replay with a different resolved destination', () => {
    const owner = new HandoffRouting('A', undefined, false);
    const input = state(owner.start());
    owner.finalize([command()], input, config);
    expect(() => owner.finalize([command('C')], input, config)).toThrow(
      'replay changed'
    );
  });

  it('merges parallel updates commutatively without duplicate transitions', () => {
    const owner = new HandoffRouting('A', undefined, false);
    const input = state(owner.start());
    owner.finalize([command()], input, config);
    const left = owner.snapshot();
    owner.finalize([command('C')], input, {
      configurable: { checkpoint_ns: 'other-task' },
    });
    const right = { ...owner.snapshot(), parallel: true };
    expect(mergeHandoffState(left, right)).toEqual(
      mergeHandoffState(right, left)
    );
    expect(mergeHandoffState(left, right)?.transitions).toHaveLength(2);
    expect(() =>
      mergeHandoffState(left, { ...right, executionId: 'different' })
    ).toThrow('different logical turns');
  });

  it('does not promote or silently grant a budget to legacy resumes', () => {
    const owner = new HandoffRouting('A', undefined, false);
    owner.resume(undefined);
    owner.finalize([command()], state(owner.snapshot()), config);
    expect(owner.outcome()).toMatchObject({
      status: 'incomplete',
      reason: 'legacy_checkpoint',
    });
    expect(() => new HandoffRouting('A', 1, false).resume(undefined)).toThrow(
      'legacy checkpoint'
    );
  });

  it('rejects incompatible entry, budget and checkpoint version', () => {
    const snapshot = new HandoffRouting('A', 2, false).snapshot();
    expect(() => new HandoffRouting('B', 2, false).resume(snapshot)).toThrow(
      'entry or budget'
    );
    expect(() => new HandoffRouting('A', 3, false).resume(snapshot)).toThrow(
      'entry or budget'
    );
    const incompatible = { ...snapshot };
    Reflect.set(incompatible, 'version', 2);
    expect(() =>
      new HandoffRouting('A', 2, false).resume(incompatible)
    ).toThrow('version');
  });

  it('returns detached snapshots rather than writable internal state', () => {
    const owner = new HandoffRouting('A', 1, false);
    owner.finalize([command()], state(owner.start()), config);
    const snapshot = owner.snapshot();
    snapshot.transitions[0].targetAgentId = 'not-accepted';
    expect(owner.outcome()).toMatchObject({
      status: 'candidate',
      agentId: 'B',
    });
  });
});

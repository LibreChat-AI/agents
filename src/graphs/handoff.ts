import { nanoid } from 'nanoid';
import { Annotation, Command, Send } from '@langchain/langgraph';
import type { BaseChannel, OverwriteValue } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  BaseGraphState,
  HandoffOutcome,
  HandoffState,
  HandoffTransition,
} from '@/types/graph';

function isSupportedVersion(version: number): boolean {
  return version === 1;
}

function isSend(value: string | Send): value is Send {
  return typeof value !== 'string' && value.lg_name === 'Send';
}

export type HandoffRequest = Pick<
  HandoffTransition,
  'sourceAgentId' | 'targetAgentId' | 'toolCallId' | 'scope'
>;

type HandoffUpdate = Partial<BaseGraphState> & {
  handoffRequest?: HandoffRequest;
};

function sameTransition(a: HandoffTransition, b: HandoffTransition): boolean {
  return (
    a.id === b.id &&
    a.sourceAgentId === b.sourceAgentId &&
    a.targetAgentId === b.targetAgentId &&
    a.toolCallId === b.toolCallId &&
    a.scope === b.scope &&
    a.depth === b.depth
  );
}

/** Union by identity makes checkpoint replay idempotent and parallel merges commutative. */
export function mergeHandoffState(
  current: HandoffState | undefined,
  update: HandoffState | undefined
): HandoffState | undefined {
  if (update == null) return current;
  if (current == null) return update;
  if (current.executionId !== update.executionId) {
    throw new Error('Cannot merge handoffs from different logical turns');
  }
  if (
    current.entryAgentId !== update.entryAgentId ||
    current.maxHandoffs !== update.maxHandoffs
  ) {
    throw new Error('Conflicting handoff checkpoint configuration');
  }
  const transitions = new Map(
    current.transitions.map((item) => [item.id, item])
  );
  for (const transition of update.transitions) {
    const existing = transitions.get(transition.id);
    if (existing != null && !sameTransition(existing, transition)) {
      throw new Error('Conflicting replayed handoff transition');
    }
    transitions.set(transition.id, transition);
  }
  return {
    ...current,
    parallel: current.parallel || update.parallel,
    historyComplete:
      current.historyComplete !== false && update.historyComplete !== false,
    transitions: [...transitions.values()].sort(
      (a, b) => a.depth - b.depth || a.id.localeCompare(b.id)
    ),
  };
}

export function handoffStateAnnotation(): BaseChannel<
  HandoffState | undefined,
  HandoffState | OverwriteValue<HandoffState | undefined> | undefined
  > {
  return Annotation<HandoffState | undefined>({
    reducer: mergeHandoffState,
    default: () => undefined,
  });
}

export class HandoffLimitError extends Error {
  constructor(readonly limit: number) {
    super(`Agent handoff limit (${limit}) reached`);
    this.name = 'HandoffLimitError';
  }
}

/** One owner per graph, never shared with isolated child executions. */
export class HandoffRouting {
  private state: HandoffState;

  constructor(
    private readonly entryAgentId: string,
    private readonly maxHandoffs: number | undefined,
    private readonly parallel: boolean
  ) {
    this.state = this.freshState();
  }

  private freshState(): HandoffState {
    return {
      version: 1,
      executionId: nanoid(),
      entryAgentId: this.entryAgentId,
      ...(this.maxHandoffs == null ? {} : { maxHandoffs: this.maxHandoffs }),
      transitions: [],
      parallel: this.parallel,
    };
  }

  start(): HandoffState {
    this.state = this.freshState();
    return this.snapshot();
  }

  resume(state: HandoffState | undefined): void {
    if (state != null) {
      this.restore(state);
      return;
    }
    if (this.maxHandoffs != null) {
      throw new Error(
        'Cannot enforce a handoff budget on a legacy checkpoint without routing state'
      );
    }
    this.state = { ...this.state, historyComplete: false };
  }

  restore(state: HandoffState | undefined): void {
    if (state == null) return;
    if (!isSupportedVersion(state.version))
      throw new Error('Unsupported handoff checkpoint version');
    if (
      state.entryAgentId !== this.entryAgentId ||
      state.maxHandoffs !== this.maxHandoffs
    ) {
      throw new Error('Cannot resume with a different handoff entry or budget');
    }
    if (this.state.executionId !== state.executionId) {
      if (this.state.transitions.length > 0) {
        throw new Error('Cannot resume a different handoff execution');
      }
      this.state = {
        ...state,
        transitions: state.transitions.map((item) => ({ ...item })),
      };
      return;
    }
    this.state = mergeHandoffState(this.state, state)!;
  }

  snapshot(): HandoffState {
    return {
      ...this.state,
      transitions: this.state.transitions.map((item) => ({ ...item })),
    };
  }

  /** Called after tools settle but before Commands can schedule recipients. */
  finalize(
    commands: Command[],
    input: BaseGraphState,
    config: RunnableConfig
  ): void {
    this.restore(input.handoffState);
    const updates: HandoffUpdate[] = [];
    let parallel = this.state.parallel;
    for (const command of commands) {
      if (command.graph !== Command.PARENT) continue;
      const sends = Array.isArray(command.goto)
        ? command.goto.filter(isSend)
        : [];
      parallel ||= sends.length > 1;
      if (sends.length > 0) {
        for (const send of sends) updates.push(send.args as HandoffUpdate);
      } else if (command.update != null) {
        updates.push(command.update as HandoffUpdate);
      }
    }
    const accepted = new Map(
      this.state.transitions.map((item) => [item.id, item])
    );
    const requests = updates.filter((update) => update.handoffRequest != null);
    if (requests.length === 0) return;
    const depth =
      input.handoffState?.transitions.length ?? this.state.transitions.length;
    const lastMessage = input.messages.at(-1);
    for (const update of requests) {
      const request = update.handoffRequest!;
      const id = JSON.stringify([
        this.state.executionId,
        config.configurable?.checkpoint_ns ?? '',
        request.sourceAgentId,
        lastMessage?.id ?? input.messages.length,
        request.toolCallId,
      ]);
      const transition = { ...request, id, depth };
      const existing = accepted.get(id);
      if (
        existing != null &&
        (existing.targetAgentId !== transition.targetAgentId ||
          existing.scope !== transition.scope)
      )
        throw new Error('Handoff replay changed its destination or scope');
      accepted.set(id, existing ?? transition);
    }
    if (this.maxHandoffs != null && accepted.size > this.maxHandoffs) {
      throw new HandoffLimitError(this.maxHandoffs);
    }
    this.state = {
      ...this.state,
      transitions: [...accepted.values()],
      parallel,
    };
    for (const update of requests) delete update.handoffRequest;
    for (const command of commands) {
      if (command.graph !== Command.PARENT) continue;
      command.update = {
        ...(command.update as HandoffUpdate),
        handoffState: this.snapshot(),
      };
      if (!Array.isArray(command.goto)) continue;
      command.goto = command.goto.map((destination) =>
        isSend(destination)
          ? new Send(destination.node, {
            ...destination.args,
            handoffState: this.snapshot(),
          })
          : destination
      );
    }
  }

  outcome(reason?: string): HandoffOutcome {
    const state = this.snapshot();
    const base = {
      executionId: state.executionId,
      entryAgentId: state.entryAgentId,
      transitions: state.transitions,
    };
    if (reason != null) return { ...base, status: 'incomplete', reason };
    if (state.historyComplete === false)
      return { ...base, status: 'incomplete', reason: 'legacy_checkpoint' };
    const persistent = state.transitions.filter(
      (item) => item.scope === 'conversation'
    );
    if (persistent.length === 0) return { ...base, status: 'unchanged' };
    if (state.parallel) return { ...base, status: 'ambiguous' };
    const last = persistent.reduce((a, b) => (a.depth > b.depth ? a : b));
    return {
      ...base,
      status: 'candidate',
      agentId: last.targetAgentId,
      transitionId: last.id,
    };
  }
}

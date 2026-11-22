import type { BaseMessage } from '@langchain/core/messages';
import type { InjectedMessage } from '@/types/tools';

type TurnContinuationSource =
  | NonNullable<InjectedMessage['source']>
  | 'routing'
  | 'handoff';

type ContainerCallOption = { container?: string | null };

/**
 * Human messages the SDK adds inside a turn (hook contexts, skill bodies,
 * steering, routing and handoff cues). They do not start a new user turn.
 */
const TURN_CONTINUATION_SOURCES: ReadonlySet<string> =
  new Set<TurnContinuationSource>([
    'skill',
    'hook',
    'system',
    'steer',
    'routing',
    'handoff',
  ]);

/** A container id repeated back to back, as a string-appending merge produces. */
const REPEATED_CONTAINER_ID = /^(container_[A-Za-z0-9_-]+?)\1+$/;

function continuesTurn(message: BaseMessage): boolean {
  const { injected, source } = message.additional_kwargs;
  return (
    injected === true ||
    (typeof source === 'string' && TURN_CONTINUATION_SOURCES.has(source))
  );
}

function readContainerId(container: unknown): string | undefined {
  if (container == null || typeof container !== 'object') {
    return undefined;
  }
  const id: unknown = Reflect.get(container, 'id');
  if (typeof id !== 'string' || id === '') {
    return undefined;
  }
  return REPEATED_CONTAINER_ID.exec(id)?.[1] ?? id;
}

/**
 * Finds the code execution container of the current turn: the latest AI
 * message after the last human message that starts a turn. A container from
 * an earlier turn is never returned, because it may have expired and nothing
 * in an earlier turn is waiting on it.
 */
export function getCurrentTurnContainerId(
  messages: BaseMessage[]
): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    const type = message.getType();
    if (type === 'human') {
      if (continuesTurn(message)) {
        continue;
      }
      return undefined;
    }
    if (type !== 'ai') {
      continue;
    }
    const id =
      readContainerId(message.additional_kwargs.container) ??
      readContainerId(Reflect.get(message.response_metadata, 'container'));
    if (id != null) {
      return id;
    }
  }
  return undefined;
}

/**
 * Anthropic requires `container` on the follow-up request while code execution
 * still has pending tool uses (for example a client tool called in the same
 * response as `code_execution`). The id arrives on the AI message of the
 * previous request, so when the caller did not set `container`, repeat the one
 * from the current turn.
 */
export function withCurrentTurnContainer<O extends object | undefined>(
  messages: BaseMessage[],
  options: O
): O {
  if ((options as ContainerCallOption | undefined)?.container != null) {
    return options;
  }
  const container = getCurrentTurnContainerId(messages);
  if (container == null) {
    return options;
  }
  return { ...options, container };
}

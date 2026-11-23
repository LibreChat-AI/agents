import type { BaseMessage } from '@langchain/core/messages';
import type { InjectedMessage } from '@/types/tools';

const IN_TURN_SOURCES: ReadonlySet<string> = new Set<
  NonNullable<InjectedMessage['source']> | 'routing' | 'handoff'
>(['skill', 'hook', 'system', 'steer', 'routing', 'handoff']);

function getContainerId(container: unknown): string | undefined {
  if (
    typeof container !== 'object' ||
    container === null ||
    !('id' in container)
  ) {
    return undefined;
  }
  return typeof container.id === 'string' && container.id !== ''
    ? container.id
    : undefined;
}

export function withCurrentTurnContainer<T extends { container?: string }>(
  messages: readonly BaseMessage[],
  options: T
): T {
  if (options.container != null) {
    return options;
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    const kwargs = message.additional_kwargs;
    const type = message.getType();
    if (type === 'human') {
      if (
        kwargs.injected !== true &&
        !(
          typeof kwargs.source === 'string' &&
          IN_TURN_SOURCES.has(kwargs.source)
        )
      ) {
        break;
      }
    }
    if (type !== 'ai') {
      continue;
    }
    const metadata = message.response_metadata as {
      model_provider?: string;
      container?: { id?: string };
    };
    if (
      metadata.model_provider != null &&
      metadata.model_provider !== 'anthropic'
    ) {
      continue;
    }
    const container =
      getContainerId(kwargs.container) ?? getContainerId(metadata.container);
    if (container != null) {
      return { ...options, container };
    }
  }
  return options;
}

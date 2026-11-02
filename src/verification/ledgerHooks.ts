import type { BaseMessage } from '@langchain/core/messages';
import type {
  BaseHookInput,
  PermissionDeniedHookOutput,
  PostToolUseFailureHookOutput,
  PostToolUseHookOutput,
  PreToolUseHookOutput,
  StopHookOutput,
} from '@/hooks/types';
import type { HookRegistry } from '@/hooks/HookRegistry';
import type { RunLedger } from './types';

/** Serialises a tool input or output for blob storage. */
function serialise(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    // JSON.stringify returns undefined for undefined and functions despite its declared type.
    const json = JSON.stringify(value) as string | undefined;
    return json ?? String(value);
  } catch {
    return String(value);
  }
}

/** Text of the last assistant message, which is what a Stop hook's run is finishing with. */
export function finalAssistantText(messages: BaseMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.getType() !== 'ai') continue;
    const { content } = message;
    if (typeof content === 'string') return content;
    const text = content
      .map((part) =>
        typeof part === 'object' &&
        'text' in part &&
        typeof part.text === 'string'
          ? part.text
          : ''
      )
      .join('');
    return text;
  }
  return null;
}

/**
 * Records every tool call, result, failure, denial and stop request of a run
 * in `ledger`. Fails closed: when the ledger cannot record a tool call, the
 * call is denied instead of running unrecorded. Hook inputs carry no timing,
 * so durations are measured between `PreToolUse` and the matching result.
 *
 * Returns a function that unregisters every hook.
 */
export function registerLedgerHooks(
  registry: HookRegistry,
  ledger: RunLedger
): () => void {
  const started = new Map<string, number>();
  const key = (input: BaseHookInput & { toolUseId: string }): string =>
    JSON.stringify([
      input.runId,
      input.executionContext?.hookSessionId,
      input.agentId,
      input.executingAgentId,
      input.toolUseId,
    ]);
  const durationOf = (
    input: BaseHookInput & { toolUseId: string }
  ): number | undefined => {
    const id = key(input);
    const at = started.get(id);
    started.delete(id);
    return at === undefined ? undefined : Date.now() - at;
  };
  /** Result rows are best effort: a failed write leaves the ledger unhealthy, which the completion gate refuses. */
  const record = (write: () => void): void => {
    try {
      write();
    } catch {
      // ledger.healthy is now false
    }
  };

  const unregister = [
    registry.register('PreToolUse', {
      internal: true,
      hooks: [
        (input): PreToolUseHookOutput => {
          try {
            ledger.append(
              {
                runId: input.runId,
                kind: 'tool_call',
                actor: 'model',
                // `agentId` is set only inside a subagent scope; `executingAgentId` names the node's agent.
                metadata: {
                  toolName: input.toolName,
                  toolUseId: input.toolUseId,
                  agentId: input.agentId,
                  executingAgentId: input.executingAgentId,
                },
              },
              { label: 'tool.input', content: serialise(input.toolInput) }
            );
            const id = key(input);
            if (!started.has(id)) started.set(id, Date.now());
            return {};
          } catch (error) {
            return {
              decision: 'deny',
              reason: `The run's evidence ledger is unavailable, so tools cannot run: ${error instanceof Error ? error.message : String(error)}`,
            };
          }
        },
      ],
    }),
    registry.register('PostToolUse', {
      internal: true,
      hooks: [
        (input): PostToolUseHookOutput => {
          record(() =>
            ledger.append(
              {
                runId: input.runId,
                kind: 'tool_result',
                actor: 'tool',
                metadata: {
                  toolName: input.toolName,
                  toolUseId: input.toolUseId,
                  durationMs: durationOf(input),
                },
              },
              { label: 'tool.output', content: serialise(input.toolOutput) }
            )
          );
          return {};
        },
      ],
    }),
    registry.register('PostToolUseFailure', {
      internal: true,
      hooks: [
        (input): PostToolUseFailureHookOutput => {
          record(() =>
            ledger.append(
              {
                runId: input.runId,
                kind: 'tool_failure',
                actor: 'tool',
                metadata: {
                  toolName: input.toolName,
                  toolUseId: input.toolUseId,
                  durationMs: durationOf(input),
                },
              },
              { label: 'tool.error', content: input.error }
            )
          );
          return {};
        },
      ],
    }),
    registry.register('PermissionDenied', {
      internal: true,
      hooks: [
        (input): PermissionDeniedHookOutput => {
          started.delete(key(input));
          record(() =>
            ledger.append({
              runId: input.runId,
              kind: 'permission_denied',
              actor: 'runtime',
              metadata: {
                toolName: input.toolName,
                toolUseId: input.toolUseId,
                reason: input.reason,
              },
            })
          );
          return {};
        },
      ],
    }),
    registry.register('Stop', {
      internal: true,
      hooks: [
        (input): StopHookOutput => {
          record(() =>
            ledger.append(
              {
                runId: input.runId,
                kind: 'stop_requested',
                actor: 'model',
                metadata: {
                  continuationCount: input.continuationCount,
                  stopReason: input.stopReason,
                },
              },
              {
                label: 'final.text',
                content: finalAssistantText(input.messages) ?? '',
              }
            )
          );
          return {};
        },
      ],
    }),
  ];
  return () => {
    unregister.forEach((fn) => fn());
    started.clear();
  };
}

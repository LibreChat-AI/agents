/* eslint-disable no-console */
// src/utils/events.ts
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCallsDispatchedEvent } from '@/types/stream';
import type { ToolExecuteBatchRequest } from '@/types/tools';
import type { AgentLogEvent } from '@/types/graph';
import { normalizeToolResultError, inheritToolExecuteProtection, validateToolExecuteProtection, hasRequiredToolExecuteProtection, protectToolExecuteBatch } from '@/protection/toolResult';
import { traceHostToolResults } from '@/langfuse';
import { GraphEvents } from '@/common';

/**
 * Safely dispatches a custom event and properly awaits it to avoid
 * race conditions where events are dispatched after run cleanup.
 */
export async function safeDispatchCustomEvent(
  event: string,
  payload: unknown,
  config?: RunnableConfig
): Promise<boolean | void> {
  let toolRequest: ToolExecuteBatchRequest | undefined;
  let hostView: ToolExecuteBatchRequest | undefined;
  try {
    if (event === GraphEvents.ON_TOOL_EXECUTE) {
      const request = payload as ToolExecuteBatchRequest;
      toolRequest = request;
      const exposed: ToolExecuteBatchRequest = {
        ...request,
        reject: (error): void => request.reject(hasRequiredToolExecuteProtection(request) ? normalizeToolResultError(error) : error),
        resolve: (
          results: Parameters<ToolExecuteBatchRequest['resolve']>[0]
        ): void => {
          const receivedAt = Date.now();
          const protectedResults = protectToolExecuteBatch(request, results, exposed);
          if (protectedResults == null) {
            const stamped = results.map((result) => ({ ...result, received_at: receivedAt }));
            void traceHostToolResults(request, results, config).then(
              () => request.resolve(stamped),
              () => { console.warn('Failed to record host tool execution metadata'); request.resolve(stamped); }
            );
            return;
          }
          void protectedResults.then(async (canonical) => {
            validateToolExecuteProtection(request, exposed);
            for (const result of canonical) result.received_at = receivedAt;
            try { await traceHostToolResults(request, canonical, config); }
            catch { console.warn('Failed to record host tool execution metadata'); }
            validateToolExecuteProtection(request, exposed);
            request.resolve(canonical);
          }).catch((error: unknown) => request.reject(normalizeToolResultError(error)));
        },
      };
      inheritToolExecuteProtection(request, exposed);
      hostView = exposed;
      payload = exposed;
    }
    await dispatchCustomEvent(event, payload, config);
    if (toolRequest != null && hostView != null) {
      try { validateToolExecuteProtection(toolRequest, hostView); }
      catch (error) { toolRequest.reject(normalizeToolResultError(error)); return false; }
    }
    return true;
  } catch (e) {
    if (toolRequest != null && hasRequiredToolExecuteProtection(toolRequest)) {
      toolRequest.reject(normalizeToolResultError(e));
      return false;
    }
    // Check if this is the known EventStreamCallbackHandler error
    if (
      e instanceof Error &&
      e.message.includes('handleCustomEvent: Run ID') &&
      e.message.includes('not found in run map')
    ) {
      // Suppress this specific error - it's expected during parallel execution
      // when EventStreamCallbackHandler loses track of run IDs
      // console.debug('Suppressed error dispatching custom event:', e);
      return false;
    }
    // Log other errors
    console.error('Error dispatching custom event:', e);
    return false;
  }
}

/** Builds an argument-free snapshot once per handoff, not once per result. */
export function createToolCallsDispatchedEvent(
  config: RunnableConfig,
  calls: readonly Pick<
    ToolExecuteBatchRequest['toolCalls'][number],
    'id' | 'name' | 'stepId'
  >[]
): ToolCallsDispatchedEvent {
  const runId = config.configurable?.run_id;
  return {
    dispatched_at: Date.now(),
    ...(typeof runId === 'string' ? { runId } : {}),
    toolCalls: calls.map(({ id, name, stepId }) => ({ id, name, stepId })),
  };
}

/**
 * Fire-and-forget diagnostic log event.
 * Debug-level logs are gated behind AGENT_DEBUG_LOGGING=true to avoid
 * overhead in production. Info/warn/error always flow through.
 * Pass `force: true` to bypass the env-var gate (e.g. invoke timing).
 */
export function emitAgentLog(
  config: RunnableConfig | undefined,
  level: AgentLogEvent['level'],
  scope: AgentLogEvent['scope'],
  message: string,
  data?: Record<string, unknown>,
  meta?: { runId?: string; agentId?: string },
  options?: { force?: boolean }
): void {
  if (!config) return;
  if (
    level === 'debug' &&
    !(options?.force ?? false) &&
    process.env.AGENT_DEBUG_LOGGING !== 'true'
  )
    return;
  void safeDispatchCustomEvent(
    GraphEvents.ON_AGENT_LOG,
    { level, scope, message, data, ...meta } satisfies AgentLogEvent,
    config
  );
}

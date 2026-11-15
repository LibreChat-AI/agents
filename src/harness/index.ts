import type {
  EventHandler,
  LCTool,
  ToolCallRequest,
  ToolExecuteBatchRequest,
  ToolExecuteResult,
} from '@/types';
import { GraphEvents } from '@/common';

/** Established by the authenticated host, never inferred from model arguments. */
export interface ExecutionIdentity {
  runtimeId: string;
  agentId: string;
  runId: string;
  userId: string;
  tenantId?: string;
  conversationId?: string;
}

/** A host reference selects an authorized implementation, not a URL or executable name. */
export interface HostToolDescriptor {
  definition: LCTool;
  reference: string;
}

export interface HostToolInvocation {
  reference: string;
  call: ToolCallRequest;
}

export interface HostToolRequest {
  version: 1;
  identity: Readonly<ExecutionIdentity>;
  calls: HostToolInvocation[];
  executionContext?: ToolExecuteBatchRequest['executionContext'];
  callerCapabilityProjection?: ToolExecuteBatchRequest['callerCapabilityProjection'];
}

/** Implementations authenticate the transport, authorize every reference, and validate arguments. */
export interface HostToolService {
  version: 1;
  invoke(
    request: HostToolRequest,
    signal?: AbortSignal
  ): Promise<ToolExecuteResult[]>;
}

export interface HostToolBindings {
  toolDefinitions: LCTool[];
  customHandlers: Record<string, EventHandler>;
}

/**
 * Binds model-visible schemas and host execution as one capability. The SDK's existing ToolNode
 * owns policy, approvals, edited arguments, result ordering and lifecycle events in both modes.
 * This adapter never retries execution; transports may deduplicate using execution identity and
 * tool-call identity, but must reconcile uncertain external effects rather than blindly rerun.
 */
export function createHostToolBindings(options: {
  identity: Readonly<ExecutionIdentity>;
  tools: readonly HostToolDescriptor[];
  service: HostToolService;
}): HostToolBindings {
  if (Number(options.service.version) !== 1)
    throw new Error('Unsupported host tool service version');
  const source = options.identity;
  const identity = Object.freeze({
    runtimeId: source.runtimeId,
    agentId: source.agentId,
    runId: source.runId,
    userId: source.userId,
    tenantId: source.tenantId,
    conversationId: source.conversationId,
  });
  for (const key of ['runtimeId', 'agentId', 'runId', 'userId'] as const) {
    if (identity[key].trim().length === 0)
      throw new Error('Incomplete execution identity');
  }
  const references = new Map<string, string>();
  const toolDefinitions: LCTool[] = [];
  for (const descriptor of options.tools) {
    const { definition, reference } = descriptor;
    if (
      !definition.name.trim() ||
      !reference.trim() ||
      references.has(definition.name)
    ) {
      throw new Error('Invalid or duplicate host tool descriptor');
    }
    references.set(definition.name, reference);
    toolDefinitions.push(structuredClone(definition));
  }
  const service = options.service;
  return {
    toolDefinitions,
    customHandlers: {
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: async (_event, value): Promise<void> => {
          const batch = value as ToolExecuteBatchRequest;
          try {
            batch.signal?.throwIfAborted();
            const ids = new Set<string>();
            const calls = batch.toolCalls.map((call): HostToolInvocation => {
              const reference = references.get(call.name);
              if (
                reference === undefined ||
                call.id.length === 0 ||
                ids.has(call.id)
              ) {
                throw new Error(
                  'Host tool invocation is outside the bound capability'
                );
              }
              ids.add(call.id);
              return { reference, call: structuredClone(call) };
            });
            const results = await service.invoke(
              {
                version: 1,
                identity,
                calls,
                executionContext: batch.executionContext,
                callerCapabilityProjection: batch.callerCapabilityProjection,
              },
              batch.signal
            );
            batch.signal?.throwIfAborted();
            if (results.length !== ids.size)
              throw new Error('Incomplete host tool results');
            const validStatuses = new Set(['success', 'error']);
            const byId = new Map<string, ToolExecuteResult>();
            for (const result of results) {
              if (
                !ids.has(result.toolCallId) ||
                byId.has(result.toolCallId) ||
                !validStatuses.has(result.status)
              ) {
                throw new Error('Invalid host tool result identity');
              }
              const receipt = { ...result };
              delete receipt.received_at;
              byId.set(result.toolCallId, receipt);
            }
            batch.resolve(calls.map(({ call }) => byId.get(call.id)!));
          } catch {
            batch.reject(new Error('Host tool execution failed'));
          }
        },
      },
    },
  };
}

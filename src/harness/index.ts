import type {
  EventHandler,
  LCTool,
  ToolCallRequest,
  ToolExecuteBatchRequest,
  ToolExecuteResult,
} from '@/types';
import type { ExecutionIdentity } from './identity';
import { snapshotExecutionIdentity } from './identity';
import { GraphEvents } from '@/common';
export { snapshotExecutionIdentity } from './identity';
export type { ExecutionIdentity } from './identity';

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
  /** SDK execution subject; distinct from the authenticated root binding. */
  executingAgentId?: string;
  executionContext?: ToolExecuteBatchRequest['executionContext'];
  callerCapabilityProjection?: ToolExecuteBatchRequest['callerCapabilityProjection'];
}

export interface HostToolSessionUpdate {
  toolCallId: string;
  context: Pick<
    ToolCallRequest,
    | 'codeSessionContext'
    | 'codeSessionBaselineId'
    | 'retainCodeSessionInputs'
    | 'runtimeSessionHint'
  >;
}
export interface HostToolEvents {
  onResult?: (result: ToolExecuteResult) => void;
  onSessionUpdate: (update: HostToolSessionUpdate) => void;
}

/** Implementations authenticate the transport, authorize every reference, and validate arguments. */
export interface HostToolService {
  version: 1;
  invoke(
    request: HostToolRequest,
    signal?: AbortSignal,
    events?: HostToolEvents
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
  const identity = snapshotExecutionIdentity(options.identity);
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
            let accepting = true;
            const earlyState = { invalid: false };
            const earlyIds = new Set<string>();
            const receipt = (result: ToolExecuteResult): ToolExecuteResult => {
              if (
                !ids.has(result.toolCallId) ||
                !['success', 'error'].includes(result.status)
              )
                throw new Error('Invalid host tool result identity');
              const copy = { ...result };
              delete copy.received_at;
              return copy;
            };
            const publishResult = (result: ToolExecuteResult): void => {
              if (!accepting || batch.signal?.aborted === true) return;
              try {
                const value = receipt(result);
                if (earlyIds.has(value.toolCallId)) return;
                earlyIds.add(value.toolCallId);
                batch.onResult?.(value);
              } catch {
                earlyState.invalid = true;
              }
            };
            const onResult = batch.onResult ? publishResult : undefined;
            const byCallId = new Map(calls.map(({ call }) => [call.id, call]));
            const reconcile = (
              target: ToolCallRequest,
              context: HostToolSessionUpdate['context']
            ): void => {
              for (const key of [
                'codeSessionContext',
                'codeSessionBaselineId',
                'retainCodeSessionInputs',
                'runtimeSessionHint',
              ] as const) {
                if (key in context)
                  Object.assign(target, {
                    [key]: structuredClone(context[key]),
                  });
              }
            };
            const onSessionUpdate = (update: HostToolSessionUpdate): void => {
              if (!accepting) return;
              const call = byCallId.get(update.toolCallId);
              if (!call) {
                earlyState.invalid = true;
                return;
              }
              reconcile(call, update.context);
            };
            let results: ToolExecuteResult[];
            try {
              results = await service.invoke(
                {
                  version: 1,
                  identity,
                  calls,
                  executingAgentId: batch.agentId,
                  executionContext: batch.executionContext,
                  callerCapabilityProjection: batch.callerCapabilityProjection,
                },
                batch.signal,
                { onResult, onSessionUpdate }
              );
            } finally {
              accepting = false;
              for (const original of batch.toolCalls) {
                const updated = byCallId.get(original.id);
                if (updated) reconcile(original, updated);
              }
            }
            if (earlyState.invalid)
              throw new Error('Invalid incremental host result');
            batch.signal?.throwIfAborted();
            if (results.length !== ids.size)
              throw new Error('Incomplete host tool results');
            const byId = new Map<string, ToolExecuteResult>();
            for (const result of results) {
              if (byId.has(result.toolCallId))
                throw new Error('Duplicate host tool result');
              byId.set(result.toolCallId, receipt(result));
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

export {
  createHostProviderTextService,
  createHostProviderTextProtection,
} from './providerText';
export type {
  HostProviderTextDescriptor,
  HostProviderTextRequest,
  HostProviderTextService,
} from './providerText';

export { HostChatModel, HostModelServiceError } from './model';
export type { ModelInvocationContext } from '@/llm/invocation';
export type {
  HostModelDescriptor,
  HostModelRequest,
  HostModelService,
  HostModelToolChoice,
  HostModelErrorCode,
} from './model';

export { createHostModelService, HostModelAdmission } from './modelService';
export type { HostModelLimits } from './modelService';

export { createHostFileService, HostFileAdmission } from './fileService';
export type { HostFileStore } from './fileService';
export { createHostFileBindings } from './fileBindings';
export type { HostFileBindings } from './fileBindings';
export {
  HostFileServiceError,
  hostFileDescriptorSchema,
  hostFileServiceDescriptorSchema,
  hostFileReadRequestSchema,
  hostFileStageRequestSchema,
  hostFilePublishRequestSchema,
} from './files';
export type {
  HostFileDescriptor,
  HostFileServiceDescriptor,
  HostFileService,
  HostFileReadRequest,
  HostFileStageRequest,
  HostFilePublishRequest,
  HostFileChunk,
  HostFilePublication,
  HostFileErrorCode,
} from './files';

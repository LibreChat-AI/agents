import type { OpenAIChatCompletionChunkChoice, OpenAIToolCall } from './index';
import type { Graph } from '@/graphs';
import { STREAM_LIMIT_ATTEMPT_KEY } from '@/llm/streamLimits';
import { StepTypes } from '@/common';

/** Both native and OpenAI-shaped run steps are emitted by SDK graphs. */
export interface OpenAIToolCallDeclaration {
  index?: number;
  id?: string;
  name?: string;
  args?: string | object;
  function?: { name?: string; arguments?: string | object };
}

export interface OpenAIRunStep {
  id: string;
  /** Run-step content position is not the outward tool-call index. */
  index?: number;
  stepDetails?: { type?: string; tool_calls?: OpenAIToolCallDeclaration[] };
}

export interface OpenAIRunStepDelta {
  id: string;
  delta?: {
    type?: string;
    tool_calls?: Array<{
      index?: number;
      id?: string;
      name?: string;
      args?: string;
      function?: { name?: string; arguments?: string };
    }>;
  };
}

/** Invocation scope comes from the SDK graph; never the response's content index. */
export type OpenAIToolCallGraph = Pick<Graph, 'getStepBaseKey'>;

/** Output is synchronous until a transport adapter can await graph dispatch. */
export interface OpenAIToolCallStreamConfig {
  toolCalls: Map<number, OpenAIToolCall>;
  emit?: (delta: OpenAIChatCompletionChunkChoice['delta']) => void;
  signal?: AbortSignal;
}

export interface OpenAIToolCallStream {
  /** Observe every model chunk/end, including text-only fallbacks, before projection. */
  observeModelAttempt: (
    metadata?: Record<string, unknown>,
    graph?: OpenAIToolCallGraph
  ) => void;
  onRunStep: (
    data: OpenAIRunStep,
    metadata?: Record<string, unknown>,
    graph?: OpenAIToolCallGraph
  ) => void;
  onRunStepDelta: (
    data: OpenAIRunStepDelta,
    metadata?: Record<string, unknown>,
    graph?: OpenAIToolCallGraph
  ) => void;
  finish: () => void;
  abort: () => void;
}

interface ProjectedToolCall {
  scope: Invocation;
  snapshotId?: string;
  snapshotName?: string;
  snapshotArgs?: string;
  idText: string;
  nameText: string;
  argsText: string;
}

interface Invocation {
  base: string;
  attempt: number;
  active: boolean;
  unattributableArguments: boolean;
  byProviderIndex: Map<number, ProjectedToolCall>;
}

/** Identity fields can be restated whole, extended cumulatively, or split. */
function mergeIdentity(existing: string, incoming: string): string {
  if (incoming === '' || existing === incoming) return existing;
  if (incoming.startsWith(existing)) return incoming;
  if (existing.startsWith(incoming)) return existing;
  return existing + incoming;
}

/** Reconcile whole JSON snapshots without dropping repeated incremental characters. */
function mergeArguments(existing: string, incoming: string): string {
  if (incoming === '') return existing;
  if (existing === '') return incoming;
  if (incoming === existing) {
    // A repeated object/array prefix is a snapshot, not another JSON root.
    // Single-character suffixes remain incremental (e.g. three streamed 'a's).
    const root = incoming.trimStart()[0];
    if (incoming.length > 1 && (root === '{' || root === '[')) return existing;
    try {
      JSON.parse(incoming);
      return incoming;
    } catch {
      return existing + incoming;
    }
  }
  if (incoming.startsWith(existing)) return incoming;
  if (existing.startsWith(incoming)) return existing;
  try {
    JSON.parse(existing);
    JSON.parse(incoming);
    return incoming;
  } catch {
    return existing + incoming;
  }
}

interface ToolCallStep {
  byId: Map<string, ProjectedToolCall>;
  byDeclaration: Map<string, ProjectedToolCall>;
  byProviderIndex: Map<number, ProjectedToolCall>;
  byPosition: Map<number, ProjectedToolCall>;
  calls: Set<ProjectedToolCall>;
}

export function createOpenAIToolCallStream(
  config: OpenAIToolCallStreamConfig
): OpenAIToolCallStream {
  const { toolCalls, emit, signal } = config;
  let phase: 'open' | 'emitting' | 'finished' | 'aborted' = 'open';
  const calls: ProjectedToolCall[] = [];
  const steps = new Map<string, ToolCallStep>();
  const invocations = new Map<string, Invocation>();
  const stepScopes = new Map<string, Invocation>();

  const baseKey = (
    metadata?: Record<string, unknown>,
    graph?: OpenAIToolCallGraph,
    stepId = ''
  ): string => {
    if (graph != null && metadata != null)
      return graph.getStepBaseKey(metadata);
    if (
      typeof metadata?.langgraph_node === 'string' &&
      typeof metadata.langgraph_step === 'number'
    ) {
      return JSON.stringify([
        metadata.run_id ?? '',
        metadata.thread_id ?? '',
        metadata.langgraph_node,
        metadata.langgraph_step,
        metadata.langgraph_checkpoint_ns ?? metadata.checkpoint_ns ?? '',
      ]);
    }
    return JSON.stringify(['step', stepId]);
  };

  const attemptId = (metadata?: Record<string, unknown>): number => {
    const attempt = metadata?.[STREAM_LIMIT_ATTEMPT_KEY];
    if (attempt == null) return 0;
    if (
      typeof attempt !== 'number' ||
      !Number.isSafeInteger(attempt) ||
      attempt < 0
    ) {
      abort();
      throw new Error('Invalid model attempt in agent response');
    }
    return attempt;
  };

  const resolveInvocation = (
    base: string,
    attempt: number
  ): Invocation | undefined => {
    let scope = invocations.get(base);
    if (scope != null && attempt < scope.attempt) return undefined;
    if (scope == null || attempt > scope.attempt) {
      if (scope != null) {
        scope.active = false;
        scope.byProviderIndex.clear();
        // Release failed-attempt bytes, but retain its scope tombstone for late events.
        for (const call of calls) {
          if (call.scope === scope) {
            call.idText = call.nameText = call.argsText = '';
            call.snapshotId = call.snapshotName = call.snapshotArgs = undefined;
          }
        }
      }
      scope = {
        base,
        attempt,
        active: true,
        unattributableArguments: false,
        byProviderIndex: new Map(),
      };
      invocations.set(base, scope);
    }
    return scope;
  };

  /** Freeze the graph segment for a declared step, but never share a scope across attempts. */
  const getScope = (
    stepId: string,
    metadata?: Record<string, unknown>,
    graph?: OpenAIToolCallGraph
  ): Invocation | undefined => {
    const attempt = attemptId(metadata);
    const key = JSON.stringify([stepId, attempt]);
    const known = stepScopes.get(key);
    if (known != null) return known.active ? known : undefined;
    const scope = resolveInvocation(baseKey(metadata, graph, stepId), attempt);
    if (scope != null) stepScopes.set(key, scope);
    return scope;
  };

  const requireStepId = (id: string): string => {
    if (typeof id !== 'string' || id.trim() === '') {
      abort();
      throw new Error('Tool-call events require a nonempty step ID');
    }
    return id;
  };

  const identityMatch = (
    scope: Invocation,
    id: string | undefined
  ): ProjectedToolCall | undefined => {
    if (id == null || id === '') return undefined;
    const matches = calls.filter(
      (call) => call.scope === scope && (call.idText || call.snapshotId) === id
    );
    return matches.length === 1 ? matches[0] : undefined;
  };

  const getStep = (stepId: string): ToolCallStep => {
    let step = steps.get(stepId);
    if (step === undefined) {
      step = {
        byId: new Map(),
        byDeclaration: new Map(),
        byProviderIndex: new Map(),
        byPosition: new Map(),
        calls: new Set(),
      };
      steps.set(stepId, step);
    }
    return step;
  };

  const allocate = (
    step: ToolCallStep,
    scope: Invocation
  ): ProjectedToolCall => {
    const call: ProjectedToolCall = {
      scope,
      idText: '',
      nameText: '',
      argsText: '',
    };
    calls.push(call);
    step.calls.add(call);
    return call;
  };

  const abort = (): void => {
    if (phase !== 'finished') {
      phase = 'aborted';
    }
    calls.length = 0;
    steps.clear();
    stepScopes.clear();
    invocations.clear();
  };

  const writable = (): boolean => {
    if (signal?.aborted === true) {
      abort();
    }
    return phase === 'open';
  };

  const ensureNotAborted = (): void => {
    if (signal?.aborted === true) abort();
    if (phase === 'aborted') {
      const error = new Error('Agent response aborted');
      error.name = 'AbortError';
      throw error;
    }
  };

  return {
    abort,
    observeModelAttempt: (metadata, graph): void => {
      if (!writable() || metadata?.[STREAM_LIMIT_ATTEMPT_KEY] == null) return;
      resolveInvocation(baseKey(metadata, graph), attemptId(metadata));
    },
    finish: (): void => {
      if (phase === 'finished' || phase === 'emitting') return;
      ensureNotAborted();
      try {
        if (
          [...invocations.values()].some(
            (scope) => scope.unattributableArguments
          )
        ) {
          throw new Error(
            'Unattributable tool call arguments in agent response'
          );
        }
        const ready: OpenAIToolCall[] = [];
        const ids = new Set<string>();
        for (const call of calls) {
          if (!call.scope.active) continue;
          const name = call.nameText || call.snapshotName;
          const args = call.argsText || call.snapshotArgs;
          if (name == null || name === '' || args == null || args === '') {
            throw new Error('Incomplete tool call in agent response');
          }
          try {
            JSON.parse(args);
          } catch {
            throw new Error('Invalid tool call arguments in agent response');
          }
          let id = call.idText || (call.snapshotId ?? '');
          if (id === '') id = `call_${ready.length}`;
          while (ids.has(id)) id = `${id}_${ready.length}`;
          ids.add(id);
          ready.push({
            id,
            type: 'function',
            function: { name, arguments: args },
          });
        }
        // Validation is atomic, publication is not. Block reentrancy without masking cancellation.
        phase = 'emitting';
        ready.forEach((call, index) => toolCalls.set(index, call));
        for (const [index, call] of ready.entries()) {
          ensureNotAborted();
          emit?.({
            tool_calls: [
              {
                index,
                id: call.id,
                type: 'function',
                function: { name: call.function.name, arguments: '' },
              },
            ],
          });
          ensureNotAborted();
          emit?.({
            tool_calls: [
              { index, function: { arguments: call.function.arguments } },
            ],
          });
          ensureNotAborted();
        }
        phase = 'finished';
      } catch (error) {
        // A partially written batch cannot be retried as a successful completion.
        toolCalls.clear();
        abort();
        throw error;
      } finally {
        abort();
      }
    },
    onRunStep: (data, metadata, graph): void => {
      if (!writable()) {
        return;
      }
      const details = data.stepDetails;
      if (
        details?.type !== StepTypes.TOOL_CALLS ||
        !Array.isArray(details.tool_calls)
      ) {
        return;
      }
      const stepId = requireStepId(data.id);
      const scope = getScope(stepId, metadata, graph);
      if (scope == null) return;
      const step = getStep(JSON.stringify([scope.base, scope.attempt, stepId]));
      const bindings = scope.byProviderIndex;
      for (const [position, toolCall] of details.tool_calls.entries()) {
        const key =
          toolCall.index === undefined
            ? `position:${position}`
            : `index:${toolCall.index}`;
        let call =
          (toolCall.id !== undefined && toolCall.id !== ''
            ? step.byId.get(toolCall.id)
            : undefined) ??
          step.byDeclaration.get(key) ??
          identityMatch(scope, toolCall.id);
        if (call === undefined && toolCall.index !== undefined) {
          call =
            step.byProviderIndex.get(toolCall.index) ??
            bindings.get(toolCall.index);
        }
        /** A declaration can follow its raw chunks. Positions are meaningful
         * only inside this declaring step, never across the whole response. */
        if (call === undefined) {
          call = step.byProviderIndex.get(position);
          if (
            call === undefined &&
            step.byDeclaration.size === 0 &&
            details.tool_calls.length === 1 &&
            step.calls.size === 1
          ) {
            call = step.calls.values().next().value;
          }
        }
        call ??= allocate(step, scope);
        step.byDeclaration.set(key, call);
        if (toolCall.id !== undefined && toolCall.id !== '') {
          call.snapshotId = toolCall.id;
          step.byId.set(toolCall.id, call);
        }
        call.snapshotName =
          toolCall.name ?? toolCall.function?.name ?? call.snapshotName;
        const args = toolCall.function?.arguments ?? toolCall.args;
        if (args !== undefined) {
          /** Both public snapshot fields accept raw JSON strings or objects. */
          try {
            call.snapshotArgs =
              typeof args === 'string' ? args : JSON.stringify(args);
          } catch {
            abort();
            throw new Error('Invalid tool call arguments in agent response');
          }
        }
        if (toolCall.index !== undefined) {
          step.byProviderIndex.set(toolCall.index, call);
          bindings.set(toolCall.index, call);
        } else if (details.tool_calls.length > 1) {
          step.byPosition.set(position, call);
        }
      }
    },
    onRunStepDelta: (data, metadata, graph): void => {
      if (!writable()) {
        return;
      }
      const delta = data.delta;
      if (
        delta?.type !== StepTypes.TOOL_CALLS ||
        !Array.isArray(delta.tool_calls)
      ) {
        return;
      }
      const stepId = requireStepId(data.id);
      const scope = getScope(stepId, metadata, graph);
      if (scope == null) return;
      const step = getStep(JSON.stringify([scope.base, scope.attempt, stepId]));
      const bindings = scope.byProviderIndex;
      for (const fragment of delta.tool_calls) {
        const name = fragment.name ?? fragment.function?.name ?? '';
        const args = fragment.args ?? fragment.function?.arguments ?? '';
        /** An index identifies an already-bound raw stream even if this event
         * carries only a substring of its ID. Never key identity by that suffix. */
        let call =
          fragment.index === undefined
            ? undefined
            : bindings.get(fragment.index);
        call ??=
          fragment.id !== undefined && fragment.id !== ''
            ? (step.byId.get(fragment.id) ?? identityMatch(scope, fragment.id))
            : undefined;
        call ??=
          fragment.index === undefined
            ? undefined
            : step.byPosition.get(fragment.index);
        if (
          call === undefined &&
          step.calls.size === 1 &&
          (fragment.index === undefined || step.byProviderIndex.size === 0)
        ) {
          const singleton = step.calls.values().next().value;
          /** Without an index, a different full ID starts another raw call, not
           * another substring of the old call. Split IDs need their index. */
          if (
            fragment.id === undefined ||
            fragment.id === '' ||
            fragment.index !== undefined ||
            (singleton?.snapshotId === undefined && singleton?.idText === '')
          ) {
            call = singleton;
          }
        }
        if (
          call === undefined &&
          (fragment.index !== undefined ||
            (fragment.id !== undefined && fragment.id !== '') ||
            name !== '')
        ) {
          call = allocate(step, scope);
        }
        if (call === undefined) {
          scope.unattributableArguments ||= args !== '';
          continue;
        }
        if (fragment.index !== undefined) {
          step.byProviderIndex.set(fragment.index, call);
          bindings.set(fragment.index, call);
        }
        if (fragment.id !== undefined && fragment.id !== '') {
          call.idText = mergeIdentity(call.idText, fragment.id);
          /** Partial IDs are not aliases across indexed calls: two parallel
           * calls can both begin with `call_`. Their index is authoritative. */
          if (fragment.index === undefined) {
            step.byId.set(call.idText, call);
          }
        }
        if (name !== '') {
          call.nameText = mergeIdentity(call.nameText, name);
        }
        if (args !== '') {
          call.argsText = mergeArguments(call.argsText, args);
        }
      }
    },
  };
}

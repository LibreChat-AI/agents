import type { OpenAIChatCompletionChunkChoice, OpenAIToolCall } from './index';
import type { Graph } from '@/graphs';
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
  id?: string;
  /** Run-step content position is not the outward tool-call index. */
  index?: number;
  stepDetails?: { type?: string; tool_calls?: OpenAIToolCallDeclaration[] };
}

export interface OpenAIRunStepDelta {
  id?: string;
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
  index: number;
  snapshotId?: string;
  snapshotName?: string;
  snapshotArgs?: string;
  idFragments: string[];
  nameFragments: string[];
  argFragments: string[];
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
  let phase: 'open' | 'finished' | 'aborted' = 'open';
  let unattributableArguments = false;
  const calls: ProjectedToolCall[] = [];
  const steps = new Map<string, ToolCallStep>();
  const invocations = new Map<string, Map<number, ProjectedToolCall>>();
  const stepScopes = new Map<string, string>();

  /** The SDK can send earlier parallel calls under the latest step ID. Its
   * invocation key owns segment/checkpoint transitions; cache it at declaration
   * so late events never borrow a new segment. Step isolation is the fallback. */
  const getBindings = (
    stepId: string,
    metadata?: Record<string, unknown>,
    graph?: OpenAIToolCallGraph
  ): Map<number, ProjectedToolCall> => {
    let scope = stepScopes.get(stepId);
    if (scope === undefined) {
      if (graph && metadata) {
        scope = graph.getStepBaseKey(metadata);
      } else if (
        typeof metadata?.langgraph_node === 'string' &&
        typeof metadata.langgraph_step === 'number'
      ) {
        scope = JSON.stringify([
          metadata.run_id ?? '',
          metadata.thread_id ?? '',
          metadata.langgraph_node,
          metadata.langgraph_step,
          metadata.langgraph_checkpoint_ns ?? metadata.checkpoint_ns ?? '',
        ]);
      } else {
        scope = JSON.stringify(['step', stepId]);
      }
      stepScopes.set(stepId, scope);
    }
    let bindings = invocations.get(scope);
    if (bindings === undefined) {
      bindings = new Map();
      invocations.set(scope, bindings);
    }
    return bindings;
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

  const allocate = (step: ToolCallStep): ProjectedToolCall => {
    const call: ProjectedToolCall = {
      index: calls.length,
      idFragments: [],
      nameFragments: [],
      argFragments: [],
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

  return {
    abort,
    finish: (): void => {
      if (phase === 'finished') {
        return;
      }
      if (!writable()) {
        const error = new Error('Agent response aborted');
        error.name = 'AbortError';
        throw error;
      }
      try {
        if (unattributableArguments) {
          throw new Error(
            'Unattributable tool call arguments in agent response'
          );
        }
        const ready: OpenAIToolCall[] = [];
        const ids = new Set<string>();
        /** There is no per-field name/ID seal in the graph event contract. An
         * OpenAI-compatible client freezes the name on the first outward chunk.
         * Validate and assemble every call before publishing any: text still
         * streams, but tool chunks wait for successful response completion. */
        for (const call of calls) {
          const name = call.nameFragments.join('') || call.snapshotName;
          const args = call.argFragments.length
            ? call.argFragments.join('')
            : call.snapshotArgs;
          if (
            name === undefined ||
            name === '' ||
            args === undefined ||
            args === ''
          ) {
            throw new Error('Incomplete tool call in agent response');
          }
          try {
            JSON.parse(args);
          } catch {
            throw new Error('Invalid tool call arguments in agent response');
          }
          let id = call.idFragments.join('');
          if (id === '') {
            id = call.snapshotId ?? '';
          }
          if (id === '') {
            id = `call_${call.index}`;
          }
          while (ids.has(id)) {
            id = `${id}_${call.index}`;
          }
          ids.add(id);
          ready.push({
            id,
            type: 'function',
            function: { name, arguments: args },
          });
        }
        for (const [index, call] of ready.entries()) {
          toolCalls.set(index, call);
        }
        /** Seal before invoking transport callbacks, including reentrant callers.
         * An emission failure cannot be retried into duplicate tool requests. */
        phase = 'finished';
        for (const [index, call] of ready.entries()) {
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
          emit?.({
            tool_calls: [
              { index, function: { arguments: call.function.arguments } },
            ],
          });
        }
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
      const step = getStep(data.id ?? '');
      const bindings = getBindings(data.id ?? '', metadata, graph);
      for (const [position, toolCall] of details.tool_calls.entries()) {
        const key =
          toolCall.index === undefined
            ? `position:${position}`
            : `index:${toolCall.index}`;
        let call =
          (toolCall.id !== undefined && toolCall.id !== ''
            ? step.byId.get(toolCall.id)
            : undefined) ?? step.byDeclaration.get(key);
        if (call === undefined && toolCall.index !== undefined) {
          call = step.byProviderIndex.get(toolCall.index);
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
        call ??= allocate(step);
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
      const step = getStep(data.id ?? '');
      const bindings = getBindings(data.id ?? '', metadata, graph);
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
            ? step.byId.get(fragment.id)
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
            (singleton?.snapshotId === undefined &&
              singleton?.idFragments.length === 0)
          ) {
            call = singleton;
          }
        }
        if (
          call === undefined &&
          ((fragment.id !== undefined && fragment.id !== '') || name !== '')
        ) {
          call = allocate(step);
        }
        if (call === undefined) {
          unattributableArguments ||= args !== '';
          continue;
        }
        if (fragment.index !== undefined) {
          step.byProviderIndex.set(fragment.index, call);
          bindings.set(fragment.index, call);
        }
        if (fragment.id !== undefined && fragment.id !== '') {
          call.idFragments.push(fragment.id);
          /** Partial IDs are not aliases across indexed calls: two parallel
           * calls can both begin with `call_`. Their index is authoritative. */
          if (fragment.index === undefined) {
            step.byId.set(call.idFragments.join(''), call);
          }
        }
        if (name !== '') {
          call.nameFragments.push(name);
        }
        if (args !== '') {
          call.argFragments.push(args);
        }
      }
    },
  };
}

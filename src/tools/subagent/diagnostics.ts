import {
  SubagentSettlementBindingError,
  SubagentDefinitionBindingError,
  SubagentInvocationBindingError,
  SubagentExecutionInvalidatedError,
} from './SubagentExecutionRegistry';
import { describeCodeApiError } from '@/tools/diagnostics';

/** Which step of child start-up failed: execution identity, then host config resolution. */
export type SubagentResolutionPhase = 'identity' | 'config';

const SUBAGENT_ERROR_TYPES = [
  ['SubagentExecutionInvalidatedError', SubagentExecutionInvalidatedError],
  ['SubagentDefinitionBindingError', SubagentDefinitionBindingError],
  ['SubagentInvocationBindingError', SubagentInvocationBindingError],
  ['SubagentSettlementBindingError', SubagentSettlementBindingError],
] as const;

type SubagentErrorLabel =
  | (typeof SUBAGENT_ERROR_TYPES)[number][0]
  | ReturnType<typeof describeCodeApiError>['type'];

const SUBAGENT_RESOLUTION_MESSAGES = {
  workspace_unavailable:
    'Subagent error: Workspace unavailable. Retry later or choose another workspace.',
  agent_unavailable: 'Subagent error: Agent not found or not accessible.',
  model_unavailable: 'Subagent error: Model or provider unavailable.',
  configuration_changed:
    'Subagent error: Subagent configuration changed. Start a new execution.',
  unknown: 'Subagent error: Unable to initialize the selected subagent.',
} as const;

/** Host classifications accepted at the public failure boundary. */
export type SubagentResolutionCause = keyof typeof SUBAGENT_RESOLUTION_MESSAGES;

export type SubagentResolutionContext = {
  parentRunId: string;
  parentAgentId?: string;
  parentToolCallId?: string;
  threadId?: string;
  childRunId: string;
  childThreadId: string;
  taskId?: string;
};

/** No received error text, writable error names, or stacks enter this payload. */
export type SubagentResolutionDiagnostic = {
  phase: SubagentResolutionPhase;
  subagentType: string;
  aborted: boolean;
  type: SubagentErrorLabel;
  cause: SubagentResolutionCause;
  message: (typeof SUBAGENT_RESOLUTION_MESSAGES)[SubagentResolutionCause];
} & Partial<SubagentResolutionContext>;

/** The original error is private. Return a classification, never error text. */
export type SubagentResolutionFailureHandler = (
  detail: Readonly<SubagentResolutionDiagnostic>,
  error: unknown
) => SubagentResolutionCause | void;

function normalizeResolutionCause(
  cause: SubagentResolutionCause | void
): SubagentResolutionCause {
  switch (cause) {
  case 'workspace_unavailable':
  case 'agent_unavailable':
  case 'model_unavailable':
  case 'configuration_changed':
    return cause;
  default:
    return 'unknown';
  }
}

export function getSubagentResolutionFailureMessage(
  cause: SubagentResolutionCause
): SubagentResolutionDiagnostic['message'] {
  return SUBAGENT_RESOLUTION_MESSAGES[normalizeResolutionCause(cause)];
}

/** Safe typed failure retained when detached execution crosses the host boundary. */
export class SubagentResolutionError extends Error {
  readonly phase: SubagentResolutionPhase;
  readonly resolutionCause: SubagentResolutionCause;

  constructor(phase: SubagentResolutionPhase, cause: SubagentResolutionCause) {
    super(getSubagentResolutionFailureMessage(cause));
    this.name = 'SubagentResolutionError';
    this.phase = phase;
    this.resolutionCause = normalizeResolutionCause(cause);
  }
}

function describeSubagentError(error: unknown): SubagentErrorLabel {
  try {
    for (const [label, constructor] of SUBAGENT_ERROR_TYPES) {
      if (error instanceof constructor) {
        return label;
      }
    }
  } catch {
    return 'UndescribableError';
  }
  return describeCodeApiError(error).type;
}

export function logSubagentResolutionFailure(
  phase: SubagentResolutionPhase,
  subagentType: string,
  signal: AbortSignal,
  error: unknown,
  context?: SubagentResolutionContext,
  onResolutionFailure?: SubagentResolutionFailureHandler
): SubagentResolutionDiagnostic {
  const detail: SubagentResolutionDiagnostic = {
    ...context,
    phase,
    subagentType,
    aborted: signal.aborted,
    type: describeSubagentError(error),
    cause: 'unknown',
    message: SUBAGENT_RESOLUTION_MESSAGES.unknown,
  };
  if (onResolutionFailure != null) {
    try {
      const cause = normalizeResolutionCause(
        onResolutionFailure(Object.freeze(detail), error)
      );
      return {
        ...detail,
        cause,
        message: getSubagentResolutionFailureMessage(cause),
      };
    } catch {
      // A broken diagnostic sink must not replace or expose the original failure.
    }
  }
  // eslint-disable-next-line no-console
  console.warn('[SubagentExecutor] Subagent resolution failed', detail);
  return detail;
}

import type { SubagentHostArgs } from '@/types/graph';
import {
  SubagentSettlementBindingError,
  SubagentDefinitionBindingError,
  SubagentInvocationBindingError,
  SubagentExecutionInvalidatedError,
} from './SubagentExecutionRegistry';
import { describeCodeApiError } from '@/tools/diagnostics';
import { isSubagentHostArgName } from './hostArgs';

/** Which step of child start-up failed: execution identity, then host config resolution. */
export type SubagentResolutionPhase = 'identity' | 'config';

/** Why a host resolver refused a declared host argument value. */
export type SubagentHostArgumentRejection = 'unavailable' | 'not_allowed';

/**
 * Thrown by a lazy resolver to refuse one declared host argument value. The
 * parent model receives a fixed message naming the argument; the value and
 * any error text stay private.
 */
export class SubagentHostArgumentError extends Error {
  readonly argument: string;
  readonly rejection: SubagentHostArgumentRejection;

  constructor(argument: string, rejection: SubagentHostArgumentRejection) {
    super('Subagent host argument was rejected.');
    this.name = 'SubagentHostArgumentError';
    this.argument = argument;
    this.rejection = rejection;
  }
}

/** Safe host-argument refusal retained across the detached delivery boundary. */
export type SubagentHostArgumentFailure = {
  argument: string;
  rejection: SubagentHostArgumentRejection;
};

const SUBAGENT_ERROR_TYPES = [
  ['SubagentHostArgumentError', SubagentHostArgumentError],
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
  host_argument_rejected:
    'Subagent error: A requested subagent argument is not available. Omit it to let the host choose, or pass a different value.',
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
  case 'host_argument_rejected':
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

/** Returns a resolver's typed host-argument refusal when its name is safe to show. */
export function getSubagentHostArgumentFailure(
  error: unknown,
  suppliedHostArgs: SubagentHostArgs | undefined
): SubagentHostArgumentFailure | undefined {
  try {
    if (
      !(error instanceof SubagentHostArgumentError) ||
      suppliedHostArgs == null
    ) {
      return undefined;
    }
    const { argument, rejection } = error;
    if (
      typeof argument !== 'string' ||
      !isSubagentHostArgName(argument) ||
      !Object.prototype.hasOwnProperty.call(suppliedHostArgs, argument)
    ) {
      return undefined;
    }
    return {
      argument,
      rejection: rejection === 'unavailable' ? 'unavailable' : 'not_allowed',
    };
  } catch {
    return undefined;
  }
}

/** Model-facing message for a refused host argument. */
export function getSubagentHostArgumentFailureMessage(
  failure: SubagentHostArgumentFailure
): string {
  const { argument } = failure;
  const omit = `Omit "${argument}" to let the host choose, or pass a different value.`;
  return failure.rejection === 'unavailable'
    ? `Subagent error: The requested "${argument}" is unavailable right now. ${omit}`
    : `Subagent error: The requested "${argument}" is not allowed for this subagent. ${omit}`;
}

/** Safe typed failure retained when detached execution crosses the host boundary. */
export class SubagentResolutionError extends Error {
  readonly phase: SubagentResolutionPhase;
  readonly resolutionCause: SubagentResolutionCause;
  readonly hostArgument?: SubagentHostArgumentFailure;

  constructor(
    phase: SubagentResolutionPhase,
    cause: SubagentResolutionCause,
    hostArgument?: SubagentHostArgumentFailure
  ) {
    super(
      hostArgument == null
        ? getSubagentResolutionFailureMessage(cause)
        : getSubagentHostArgumentFailureMessage(hostArgument)
    );
    this.name = 'SubagentResolutionError';
    this.phase = phase;
    this.resolutionCause = normalizeResolutionCause(cause);
    if (hostArgument != null) {
      this.hostArgument = hostArgument;
    }
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
  onResolutionFailure?: SubagentResolutionFailureHandler,
  suppliedHostArgs?: SubagentHostArgs
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
  const hostArgumentRejected =
    getSubagentHostArgumentFailure(error, suppliedHostArgs) != null;
  if (hostArgumentRejected) {
    detail.cause = 'host_argument_rejected';
    detail.message = SUBAGENT_RESOLUTION_MESSAGES.host_argument_rejected;
  }
  if (onResolutionFailure != null) {
    try {
      const reported = onResolutionFailure(Object.freeze({ ...detail }), error);
      const cause = hostArgumentRejected
        ? detail.cause
        : normalizeResolutionCause(reported);
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

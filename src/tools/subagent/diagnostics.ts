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

/**
 * Follows the rule in `@/tools/diagnostics`: every field is a value this
 * module owns or a host-registered subagent type, never text from the error.
 * The failure the model sees is a fixed sentence, so this line is the only
 * operator account of it; the host logs its own error text where it throws.
 */
export type SubagentResolutionDiagnostic = {
  phase: SubagentResolutionPhase;
  subagentType: string;
  aborted: boolean;
  type: SubagentErrorLabel;
};

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
  error: unknown
): void {
  const detail: SubagentResolutionDiagnostic = {
    phase,
    subagentType,
    aborted: signal.aborted,
    type: describeSubagentError(error),
  };
  // eslint-disable-next-line no-console
  console.warn('[SubagentExecutor] Subagent resolution failed', detail);
}

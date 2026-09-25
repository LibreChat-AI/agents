import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { truncateToolResultContent } from '@/utils/truncation';
import { StreamLimitExceededError } from '@/llm/streamLimits';

export const STOPPED_RUN_TOOL_ERROR =
  'STOP. The user doesn\'t want to proceed with this tool use. ' +
  'The run was stopped; the tool may not have completed, and its effects should not be assumed. ' +
  'STOP what you are doing and wait for the user to tell you how to proceed.';

/** A host result only carries error text, so require the run signal as well. */
function isStoppedRunToolError(
  message: string | undefined,
  signal?: AbortSignal,
  error?: Error
): boolean {
  if (
    signal?.aborted !== true ||
    signal.reason instanceof StreamLimitExceededError ||
    signal.reason instanceof PreparedSubagentError
  ) {
    return false;
  }
  if (
    (error != null && error === signal.reason) ||
    (error instanceof Error &&
      (error.name === 'AbortError' ||
        ('code' in error &&
          (error.code === 'ABORT_ERR' || error.code === 'ERR_CANCELED'))))
  ) {
    return true;
  }
  return (
    message != null &&
    /AbortError|(?:operation|request|stream) was aborted/i.test(message)
  );
}

export function formatToolErrorContent(
  message: string | undefined,
  maxChars: number,
  signal?: AbortSignal,
  error?: Error
): string {
  const content = isStoppedRunToolError(message, signal, error)
    ? STOPPED_RUN_TOOL_ERROR
    : `Error: ${message ?? 'Unknown error'}\n Please fix your mistakes.`;
  return truncateToolResultContent(content, maxChars);
}

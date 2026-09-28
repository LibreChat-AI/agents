import { describe, it, expect } from '@jest/globals';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import {
  STOPPED_RUN_TOOL_ERROR,
  formatToolErrorContent,
} from '@/tools/toolErrorContent';

const maxChars = 1000;

describe('formatToolErrorContent', () => {
  it('instructs the model to stop for LibreChat host abort results on a stopped run', () => {
    const controller = new AbortController();
    controller.abort();

    expect(
      formatToolErrorContent(
        'MCP error -32001: AbortError: This operation was aborted',
        maxChars,
        controller.signal
      )
    ).toBe(STOPPED_RUN_TOOL_ERROR);
    expect(
      formatToolErrorContent(
        'This operation was aborted',
        maxChars,
        controller.signal
      )
    ).toBe(STOPPED_RUN_TOOL_ERROR);
  });

  it('preserves ordinary errors, even when they race with a stop', () => {
    const controller = new AbortController();
    controller.abort();
    expect(
      formatToolErrorContent('Permission denied', maxChars, controller.signal)
    ).toBe('Error: Permission denied\n Please fix your mistakes.');
    expect(
      formatToolErrorContent('AbortError: remote timeout', maxChars)
    ).toBe('Error: AbortError: remote timeout\n Please fix your mistakes.');
  });

  it('recognizes direct-tool abort errors and the owning signal reason', () => {
    const controller = new AbortController();
    const error = new Error('User cancelled the run');
    controller.abort(error);

    expect(
      formatToolErrorContent(error.message, maxChars, controller.signal, error)
    ).toBe(STOPPED_RUN_TOOL_ERROR);

    const other = new Error('Tool cancelled');
    other.name = 'AbortError';
    expect(
      formatToolErrorContent(other.message, maxChars, controller.signal, other)
    ).toBe(STOPPED_RUN_TOOL_ERROR);
  });

  it('does not attribute circuit-breaker safety aborts to a user stop', () => {
    const breaker = new AbortController();
    breaker.abort(
      new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit: 10,
        observed: 11,
        toolName: 'db_query',
      })
    );
    expect(
      formatToolErrorContent('AbortError', maxChars, breaker.signal)
    ).toBe('Error: AbortError\n Please fix your mistakes.');

    const preparation = new AbortController();
    preparation.abort(new PreparedSubagentError('child run failed'));
    expect(
      formatToolErrorContent('AbortError', maxChars, preparation.signal)
    ).toBe('Error: AbortError\n Please fix your mistakes.');
  });
});

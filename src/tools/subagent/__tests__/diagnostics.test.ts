import { describe, expect, it, jest, afterEach } from '@jest/globals';
import type { SubagentResolutionFailureHandler } from '../diagnostics';
import {
  getSubagentResolutionFailureMessage,
  logSubagentResolutionFailure,
  SubagentResolutionError,
} from '../diagnostics';
import {
  SubagentDefinitionBindingError,
  SubagentExecutionInvalidatedError,
} from '../SubagentExecutionRegistry';

const signal = new AbortController().signal;
const secret =
  'oauth-token=private-secret\n    at private-stack (secret.ts:42)';

function silenceConsole() {
  return jest.spyOn(console, 'warn').mockImplementation(() => {});
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe('subagent resolution diagnostics', () => {
  it.each([
    ['workspace_unavailable', 'Workspace unavailable'],
    ['agent_unavailable', 'Agent not found or not accessible'],
    ['model_unavailable', 'Model or provider unavailable'],
    ['configuration_changed', 'Subagent configuration changed'],
    ['unknown', 'Unable to initialize the selected subagent'],
  ] as const)('accepts only safe public text for %s', (cause, message) => {
    const warn = silenceConsole();
    const error = new Error(secret);
    error.name = secret;
    error.stack = secret;
    const hook = jest.fn<SubagentResolutionFailureHandler>(() => cause);
    const detail = logSubagentResolutionFailure(
      'config',
      'reviewer',
      signal,
      error,
      undefined,
      hook
    );

    expect(hook).toHaveBeenCalledWith(
      expect.objectContaining({
        phase: 'config',
        subagentType: 'reviewer',
        aborted: false,
        type: 'Error',
      }),
      error
    );
    expect(detail).toMatchObject({
      cause,
      message: getSubagentResolutionFailureMessage(cause),
    });
    expect(detail.message).toContain(message);
    expect(JSON.stringify(detail)).not.toContain('private');
    expect(JSON.stringify(detail)).not.toContain('oauth-token');
    expect(detail).not.toHaveProperty('stack');
    expect(warn).not.toHaveBeenCalled();

    const publicError = new SubagentResolutionError('config', cause);
    expect(publicError.message).toBe(detail.message);
    expect(publicError).toMatchObject({
      phase: 'config',
      resolutionCause: cause,
    });
    expect(publicError.stack).not.toContain('private');
    expect(publicError).not.toHaveProperty('cause');
  });

  it('falls back to console without quoting error properties', () => {
    const warn = silenceConsole();
    const error = new Error(secret);
    for (const property of ['message', 'name', 'stack']) {
      Object.defineProperty(error, property, {
        get: () => {
          throw new Error(secret);
        },
      });
    }
    const detail = logSubagentResolutionFailure(
      'identity',
      'reviewer',
      signal,
      error
    );

    expect(detail).toMatchObject({
      phase: 'identity',
      type: 'Error',
      cause: 'unknown',
    });
    expect(warn).toHaveBeenCalledWith(
      '[SubagentExecutor] Subagent resolution failed',
      detail
    );
    expect(JSON.stringify(warn.mock.calls)).not.toContain('private');
  });

  it('falls back safely when the host sink throws', () => {
    const warn = silenceConsole();
    const detail = logSubagentResolutionFailure(
      'config',
      'reviewer',
      signal,
      new Error(secret),
      undefined,
      () => {
        throw new Error(secret);
      }
    );

    expect(detail.cause).toBe('unknown');
    expect(warn).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(warn.mock.calls)).not.toContain('private');
  });

  it('does not let the host mutate the public failure payload', () => {
    const warn = silenceConsole();
    const detail = logSubagentResolutionFailure(
      'config',
      'reviewer',
      signal,
      new Error(secret),
      undefined,
      (input) => {
        Object.assign(input, { message: secret, cause: secret });
        return 'workspace_unavailable';
      }
    );

    expect(detail.cause).toBe('unknown');
    expect(JSON.stringify(warn.mock.calls)).not.toContain('private');
  });

  it.each(['constructor', '__proto__', secret, { message: secret }])(
    'rejects an invalid runtime classification: %p',
    (cause) => {
      silenceConsole();
      const hook = (() => cause) as SubagentResolutionFailureHandler;
      const detail = logSubagentResolutionFailure(
        'config',
        'reviewer',
        signal,
        new Error(secret),
        undefined,
        hook
      );

      expect(detail.cause).toBe('unknown');
      expect(detail.message).toBe(
        getSubagentResolutionFailureMessage('unknown')
      );
      expect(JSON.stringify(detail)).not.toContain('private');
    }
  );

  it.each([
    [new SubagentDefinitionBindingError(), 'SubagentDefinitionBindingError'],
    [
      new SubagentExecutionInvalidatedError(),
      'SubagentExecutionInvalidatedError',
    ],
    [new TypeError(secret), 'TypeError'],
    [secret, 'string'],
    [undefined, 'undefined'],
    [
      new Proxy(
        {},
        {
          getPrototypeOf: () => {
            throw new Error(secret);
          },
        }
      ),
      'UndescribableError',
    ],
  ] as const)('retains a safe error classification', (error, type) => {
    silenceConsole();
    expect(
      logSubagentResolutionFailure('config', 'reviewer', signal, error).type
    ).toBe(type);
  });
});

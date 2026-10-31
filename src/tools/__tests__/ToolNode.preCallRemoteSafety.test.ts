import { z } from 'zod';
import { createServer } from 'node:http';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { AddressInfo } from 'node:net';
import type { ToolResultProtection } from '@/protection/toolResult';
import type { LCTool, PTCToolResult } from '@/types';
import { createBashProgrammaticToolCallingTool } from '@/tools/BashProgrammaticToolCalling';
import { createProgrammaticToolCallingTool } from '@/tools/ProgrammaticToolCalling';
import { createProviderTextRestartCancellation } from '@/protection/providerText';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { detachToolSafetyError } from '@/protection/toolResult';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { ToolNode } from '@/tools/ToolNode';

const RAW = 'ir35.ir36.raw@example.invalid';
const SAFE = 'Approved diagnostic';
type Runtime = 'python' | 'bash';
type SafetyKind = 'prepared' | 'stream';
type ProtectionMode = 'selected' | 'absent' | 'unselected';

function policy(inspected: string[] = []): ToolResultProtection {
  return {
    version: 1,
    toolNames: ['lookup'],
    timeoutMs: 10000,
    maxAttemptBytes: 65536,
    maxBufferedBytes: 262144,
    classify: () => 'prose',
    inspect: ({ content }) => {
      inspected.push(content);
      return {
        version: 1,
        ok: true,
        value: { content: SAFE, replacements: 1, categories: [] },
      };
    },
  };
}

function source(fn: () => string | Promise<string>): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: 'lookup',
    description: 'Focused safety control.',
    schema: z.object({ count: z.number({ error: RAW }) }),
    func: async () => fn(),
  });
}

function state(
  name = 'lookup',
  args: { count: string | number } = { count: 42 }
) {
  return {
    messages: [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'pre-call-control', name, args, type: 'tool_call' }],
      }),
    ],
  };
}

function safetyError(
  kind: SafetyKind
): PreparedSubagentError | StreamLimitExceededError {
  const error =
    kind === 'prepared'
      ? new PreparedSubagentError(RAW, { cause: new Error(RAW) })
      : new StreamLimitExceededError({
        kind: 'delta_events',
        limit: 10,
        observed: 11,
        toolName: RAW,
      });
  error.message = RAW;
  error.stack = RAW;
  Object.defineProperty(error, 'raw', { value: RAW, enumerable: true });
  return error;
}

function assertSafeError(error: Error): void {
  expect(String(error)).not.toContain(RAW);
  expect(error).not.toHaveProperty('cause');
  expect(error).not.toHaveProperty('raw');
  for (const key of Reflect.ownKeys(error)) {
    const descriptor = Object.getOwnPropertyDescriptor(error, key);
    if (typeof descriptor?.value === 'string')
      expect(descriptor.value).not.toContain(RAW);
  }
}

const definition: LCTool = {
  name: 'lookup',
  allowed_callers: ['code_execution'],
  parameters: { type: 'object', properties: { count: { type: 'number' } } },
};

async function remoteFixture(
  runtime: Runtime,
  lookup: DynamicStructuredTool,
  mode: ProtectionMode = 'selected'
) {
  const requests: string[] = [];
  const server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk: Buffer): void => {
      body += chunk.toString();
    });
    request.on('end', (): void => {
      requests.push(body);
      response.setHeader('Content-Type', 'application/json');
      response.end(
        JSON.stringify(
          requests.length === 1
            ? {
              status: 'tool_call_required',
              continuation_token: 'remote-safety-control',
              tool_calls: [
                {
                  id: 'nested-safety-control',
                  name: 'lookup',
                  input: { count: 42 },
                },
              ],
            }
            : { status: 'completed', stdout: 'Allowed completion', files: [] }
        )
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const runner =
    runtime === 'python'
      ? createProgrammaticToolCallingTool({ baseUrl })
      : createBashProgrammaticToolCallingTool({ baseUrl });
  const selectedPolicy = {
    ...policy(),
    toolNames: mode === 'unselected' ? ['other'] : ['lookup'],
  };
  const protection = mode === 'absent' ? undefined : selectedPolicy;
  const args = {
    code:
      runtime === 'python'
        ? 'print(await lookup(count=42))'
        : 'lookup \'{"count":42}\'',
    tool_manifest: ['lookup'],
  };
  return {
    requests,
    runner,
    protection,
    args,
    context: {
      toolCall: {
        id: 'remote-safety-control',
        name: runner.name,
        type: 'tool_call' as const,
        args: {},
        toolMap: new Map([['lookup', lookup]]),
        toolDefs: [definition],
        toolResultProtection: protection,
      },
    },
    close: async (): Promise<void> => {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

it.each(
  (['object', 'array'] as const).flatMap((shape) =>
    [false, true].flatMap((withSignal) =>
      (
        ['message', 'accessor', 'stack', 'cause', 'alias', 'control'] as const
      ).map((mutation) => [shape, withSignal, mutation] as const)
    )
  )
)(
  'IR35 readmits pre-call schema failures after native chain callbacks (%s/signal=%s/%s)',
  async (shape, withSignal, mutation) => {
    let executions = 0;
    let reads = 0;
    let callbacks = 0;
    let retained: Error | undefined;
    const inspected: string[] = [];
    const observed: string[] = [];
    const callback = BaseCallbackHandler.fromMethods({
      handleChainError: (error): void => {
        callbacks++;
        retained = error;
        observed.push(error.message);
        if (mutation === 'message') error.message = RAW;
        if (mutation === 'accessor')
          Object.defineProperty(error, 'message', {
            get: () => {
              reads++;
              throw new Error(RAW);
            },
          });
        if (mutation === 'stack') error.stack = RAW;
        if (mutation === 'cause')
          Object.defineProperty(error, 'cause', { value: new Error(RAW) });
        if (mutation === 'alias')
          Object.defineProperty(error, Symbol('raw-alias'), { value: RAW });
      },
    });
    callback.awaitHandlers = true;
    const node = new ToolNode({
      trace: true,
      tools: [
        source(() => {
          executions++;
          return 'Must not execute';
        }),
      ],
      toolResultProtection: policy(inspected),
      handleToolErrors: false,
    });
    const input = state('lookup', { count: RAW });
    const error = (await node
      .invoke(shape === 'object' ? input : input.messages, {
        callbacks: [callback],
        ...(withSignal ? { signal: new AbortController().signal } : {}),
      })
      .catch((value: Error) => value)) as Error;
    expect(executions).toBe(0);
    expect(callbacks).toBe(1);
    expect(observed).toEqual([SAFE]);
    expect(inspected).toHaveLength(1);
    expect(inspected[0]).toContain(RAW);
    expect(error).not.toBe(retained);
    expect(reads).toBe(0);
    assertSafeError(error);
    if (mutation === 'message' || mutation === 'accessor')
      expect(error).toMatchObject({ code: 'unsupported' });
    else {
      expect(error.message).toBe(SAFE);
      retained!.message = RAW;
      assertSafeError(error);
    }
  }
);

it.each(
  (['python', 'bash'] as const).flatMap((runtime) =>
    (['prepared', 'stream'] as const).flatMap((kind) =>
      [false, true].map((mutate) => [runtime, kind, mutate] as const)
    )
  )
)(
  'IR36 keeps selected nested terminal errors through remote runners (%s/%s/mutation=%s)',
  async (runtime, kind, mutate) => {
    const original = safetyError(kind);
    let callbacks = 0;
    let executions = 0;
    let retained: Error | undefined;
    const observed: string[] = [];
    const callback = BaseCallbackHandler.fromMethods({
      handleToolError: (error): void => {
        callbacks++;
        retained = error;
        observed.push(String(error), JSON.stringify(error), error.stack ?? '');
        if (mutate) {
          error.message = RAW;
          error.stack = RAW;
          Object.defineProperty(error, 'cause', { value: RAW });
        }
      },
    });
    callback.awaitHandlers = true;
    const lookup = source(() => {
      executions++;
      throw original;
    });
    lookup.callbacks = [callback];
    const fixture = await remoteFixture(runtime, lookup);
    try {
      const error = (await fixture.runner
        .invoke(fixture.args, fixture.context)
        .catch((value: Error) => value)) as Error;
      expect(executions).toBe(1);
      expect(callbacks).toBe(1);
      expect(observed.join('')).not.toContain(RAW);
      expect(fixture.requests).toHaveLength(1);
      expect(error).toBeInstanceOf(
        kind === 'prepared' ? PreparedSubagentError : StreamLimitExceededError
      );
      expect(error).not.toBe(original);
      expect(error).not.toBe(retained);
      expect(detachToolSafetyError(error, true)).toBeDefined();
      assertSafeError(error);
      if (kind === 'stream')
        expect(error).toMatchObject({
          kind: 'delta_events',
          limit: 10,
          observed: 11,
        });
    } finally {
      await fixture.close();
    }
  }
);

it.each(
  (['python', 'bash'] as const).flatMap((runtime) =>
    (['prepared', 'stream'] as const).map((kind) => [runtime, kind] as const)
  )
)(
  'IR36 aborts the ToolNode instead of returning a remote tool failure (%s/%s)',
  async (runtime, kind) => {
    const lookup = source(() => {
      throw safetyError(kind);
    });
    const fixture = await remoteFixture(runtime, lookup);
    let completions = 0;
    const callback = BaseCallbackHandler.fromMethods({
      handleToolEnd: (): void => {
        completions++;
      },
    });
    callback.awaitHandlers = true;
    try {
      const node = new ToolNode({
        trace: true,
        tools: [fixture.runner, lookup],
        toolRegistry: new Map([['lookup', definition]]),
        toolResultProtection: fixture.protection,
      });
      const result = await node
        .invoke(
          {
            messages: [
              new AIMessage({
                content: '',
                tool_calls: [
                  {
                    id: 'outer-safety-control',
                    name: fixture.runner.name,
                    args: fixture.args,
                    type: 'tool_call',
                  },
                ],
              }),
            ],
          },
          { callbacks: [callback] }
        )
        .catch((error: Error) => error);
      expect(fixture.requests).toHaveLength(1);
      expect(result).toBeInstanceOf(
        kind === 'prepared' ? PreparedSubagentError : StreamLimitExceededError
      );
      expect(completions).toBe(0);
      assertSafeError(result as Error);
    } finally {
      await fixture.close();
    }
  }
);

it.each(
  (['python', 'bash'] as const).flatMap((runtime) =>
    (['prepared', 'stream'] as const).flatMap((kind) =>
      (['absent', 'unselected'] as const).map(
        (mode) => [runtime, kind, mode] as const
      )
    )
  )
)(
  'preserves legacy remote nested safety diagnostics (%s/%s/%s)',
  async (runtime, kind, mode) => {
    const original = safetyError(kind);
    const fixture = await remoteFixture(
      runtime,
      source(() => {
        throw original;
      }),
      mode
    );
    try {
      const output = await fixture.runner.invoke(fixture.args, fixture.context);
      expect(output).toMatchObject({
        content: expect.stringContaining('Allowed completion'),
      });
      expect(fixture.requests).toHaveLength(2);
      const continuation = JSON.parse(fixture.requests[1]) as {
        tool_results: PTCToolResult[];
      };
      expect(continuation.tool_results).toEqual([
        {
          call_id: 'nested-safety-control',
          result: null,
          is_error: true,
          error_message: RAW,
        },
      ]);
      expect(detachToolSafetyError(original, true)).toBeUndefined();
    } finally {
      await fixture.close();
    }
  }
);

it.each(['python', 'bash'] as const)(
  'preserves the private restart cancellation through remote %s execution',
  async (runtime) => {
    const marker = createProviderTextRestartCancellation();
    const fixture = await remoteFixture(
      runtime,
      source(() => {
        throw marker;
      })
    );
    try {
      await expect(
        fixture.runner.invoke(fixture.args, fixture.context)
      ).rejects.toBe(marker);
      expect(fixture.requests).toHaveLength(1);
    } finally {
      await fixture.close();
    }
  }
);

it.each(
  (['python', 'bash'] as const).flatMap((runtime) =>
    (['prepared', 'stream'] as const).map((kind) => [runtime, kind] as const)
  )
)(
  'does not promote callback-forged public safety classes to terminal provenance (%s/%s)',
  async (runtime, kind) => {
    const observed: string[] = [];
    const callback = BaseCallbackHandler.fromMethods({
      handleToolError: (error): void => {
        observed.push(error.message);
        Object.setPrototypeOf(
          error,
          kind === 'prepared'
            ? PreparedSubagentError.prototype
            : StreamLimitExceededError.prototype
        );
      },
    });
    callback.awaitHandlers = true;
    const lookup = source(() => {
      throw new Error(RAW);
    });
    lookup.callbacks = [callback];
    const fixture = await remoteFixture(runtime, lookup);
    try {
      const result = await fixture.runner.invoke(fixture.args, fixture.context);
      expect(result).toMatchObject({
        content: expect.stringContaining('Allowed completion'),
      });
      expect(fixture.requests).toHaveLength(2);
      const continuation = JSON.parse(fixture.requests[1]) as {
        tool_results: PTCToolResult[];
      };
      expect(continuation.tool_results[0]).toMatchObject({
        is_error: true,
        error_message: SAFE,
      });
      expect(fixture.requests[1]).not.toContain(RAW);
      expect(observed).toEqual([SAFE]);
    } finally {
      await fixture.close();
    }
  }
);

it.each(['absent', 'unselected'] as const)(
  'preserves legacy pre-call schema error identity when protection is %s',
  async (mode) => {
    let retained: Error | undefined;
    const callback = BaseCallbackHandler.fromMethods({
      handleChainError: (error): void => {
        retained = error;
      },
    });
    callback.awaitHandlers = true;
    const node = new ToolNode({
      trace: true,
      tools: [source(() => 'Must not execute')],
      handleToolErrors: false,
      ...(mode === 'unselected'
        ? { toolResultProtection: { ...policy(), toolNames: ['other'] } }
        : {}),
    });
    const error = await node
      .invoke(state('lookup', { count: RAW }), { callbacks: [callback] })
      .catch((value: Error) => value);
    expect(error).toBe(retained);
    expect((error as Error).message).toContain(RAW);
  }
);

it('still handles a selected schema error as a sanitized ToolMessage when enabled', async () => {
  let executions = 0;
  const inspected: string[] = [];
  const node = new ToolNode({
    tools: [
      source(() => {
        executions++;
        return 'Must not execute';
      }),
    ],
    toolResultProtection: policy(inspected),
  });
  const result = (await node.invoke(state('lookup', { count: RAW }))) as {
    messages: ToolMessage[];
  };
  expect(result.messages[0]).toMatchObject({
    status: 'error',
    content: expect.stringContaining(SAFE),
  });
  expect(inspected).toHaveLength(1);
  expect(executions).toBe(0);
});

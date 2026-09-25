import fetch from 'node-fetch';
import { Readable } from 'node:stream';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { RequestInit } from 'node-fetch';
import type * as t from '@/types';
import {
  createLocalProgrammaticToolCallingTool,
  createLocalBashProgrammaticToolCallingTool,
} from '../local/LocalProgrammaticToolCalling';
import {
  CodeApiRequestError,
  buildCodeApiHttpErrorMessage,
  createCodeExecutionTool,
  resolveCodeApiAuthHeaders,
} from '../CodeExecutor';
import {
  clampCodeApiRunTimeoutMs,
  createCodeApiRunTimeoutSchema,
  MAX_CODE_API_RUN_TIMEOUT_SCHEMA_MS,
} from '../ptcTimeout';
import {
  createProgrammaticToolCallingTool,
  fetchSessionFiles,
  makeRequest,
} from '../ProgrammaticToolCalling';
import { createBashProgrammaticToolCallingTool } from '../BashProgrammaticToolCalling';
import { createBashExecutionTool } from '../BashExecutor';

jest.mock('node-fetch', () => ({
  __esModule: true,
  default: jest.fn(),
}));

type FetchMock = jest.MockedFunction<
  (url: unknown, init?: unknown) => Promise<unknown>
>;

type CodeApiRequestBody = {
  code?: string;
  timeout?: number;
  continuation_token?: string;
  runtime_session_hint?: string;
  tool_results?: t.PTCToolResult[];
};

type TimeoutSchemaForTest = {
  default: number;
  maximum: number;
  description: string;
};

type ToolSchemaForTest = {
  properties: {
    timeout: TimeoutSchemaForTest;
  };
};

const fetchMock = fetch as unknown as FetchMock;

function requestBodyAt(callIndex: number): CodeApiRequestBody {
  const init = fetchMock.mock.calls[callIndex]?.[1] as RequestInit;
  return JSON.parse(init.body as string) as CodeApiRequestBody;
}

function requestHeadersAt(callIndex: number): Record<string, string> {
  const init = fetchMock.mock.calls[callIndex]?.[1] as RequestInit;
  return init.headers as Record<string, string>;
}

function timeoutSchemaForTest(toolSchema: unknown): TimeoutSchemaForTest {
  return (toolSchema as ToolSchemaForTest).properties.timeout;
}

function jsonResponse(body: unknown): unknown {
  return {
    ok: true,
    json: jest.fn(async () => body),
    text: jest.fn(async () => JSON.stringify(body)),
  };
}

function completedResponse(stdout = 'ok'): unknown {
  return jsonResponse({
    status: 'completed',
    session_id: 'session_123',
    stdout,
  });
}

function errorResponse(status: number, body: string): unknown {
  return {
    ok: false,
    status,
    text: jest.fn(async () => body),
  };
}

const toolDefs = [
  {
    name: 'lookup_user',
    description: 'Lookup a user',
    parameters: {
      type: 'object',
      properties: {},
    },
  },
] as unknown as t.LCTool[];

function toolMap(): t.ToolMap {
  return new Map([
    [
      'lookup_user',
      {
        name: 'lookup_user',
        invoke: jest.fn(async () => ({ id: 'user_123' })),
      },
    ],
  ]) as unknown as t.ToolMap;
}

describe('CodeAPI auth header injection', () => {
  let errorSpy: jest.SpiedFunction<typeof console.error>;
  let warnSpy: jest.SpiedFunction<typeof console.warn>;

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(completedResponse());
    errorSpy = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    errorSpy.mockRestore();
    warnSpy.mockRestore();
  });

  it('resolves static and dynamic auth header params', async () => {
    await expect(
      resolveCodeApiAuthHeaders({ Authorization: 'Bearer static' })
    ).resolves.toEqual({
      Authorization: 'Bearer static',
    });
    await expect(
      resolveCodeApiAuthHeaders(async () => ({
        Authorization: 'Bearer dynamic',
      }))
    ).resolves.toEqual({
      Authorization: 'Bearer dynamic',
    });
    await expect(resolveCodeApiAuthHeaders()).resolves.toEqual({});
  });

  it('keeps the no-auth request path unchanged', async () => {
    await makeRequest('https://code.example.com/exec/programmatic', {
      code: 'print(1)',
    });

    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/exec/programmatic',
      expect.objectContaining({
        headers: expect.not.objectContaining({
          Authorization: expect.any(String),
        }),
      })
    );
  });

  it('sends an authenticated explicit cancellation when a PTC request aborts', async () => {
    const controller = new AbortController();
    fetchMock
      .mockImplementationOnce((_url, rawInit) => {
        const init = rawInit as RequestInit;
        return new Promise((_resolve, reject) => {
          init.signal?.addEventListener(
            'abort',
            () => reject(new DOMException('aborted', 'AbortError')),
            { once: true }
          );
        });
      })
      .mockResolvedValueOnce(
        jsonResponse({ status: 'cancellation_requested' })
      );

    const request = makeRequest(
      'https://code.example.com/exec/programmatic',
      { code: 'sleep 30' },
      undefined,
      { Authorization: 'Bearer scoped' },
      'stateful',
      controller.signal
    ).catch((error: unknown) => error);
    await new Promise<void>((resolve) => setImmediate(resolve));
    controller.abort();
    await request;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const requestHeaders = requestHeadersAt(0);
    const requestId = requestHeaders['X-LibreChat-Code-Request-ID'];
    expect(requestId).toMatch(/^[0-9a-f-]{36}$/);
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      'https://code.example.com/exec/programmatic/cancel'
    );
    expect(requestHeadersAt(1)).toEqual(
      expect.objectContaining({
        Authorization: 'Bearer scoped',
        'X-CodeAPI-Expected-Profile': 'stateful',
      })
    );
    expect(requestBodyAt(1)).toEqual({ request_id: requestId });
    expect((fetchMock.mock.calls[1]?.[1] as RequestInit).signal).not.toBe(
      controller.signal
    );
  });

  it('maps dynamic auth-header failures to the safe authorization error', async () => {
    const authHeaders = jest.fn(async () => {
      throw new Error(
        'credential helper failed for secret codeapi-signing-key in namespace internal-auth'
      );
    });
    const tool = createProgrammaticToolCallingTool({ authHeaders });

    const error = await tool
      .invoke(
        { code: 'print("hello")' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution is not authorized. Verify access before trying again.'
    );
    expect((error as Error).message).not.toContain('Please retry');
    expect((error as Error).message).not.toContain('codeapi-signing-key');
    expect((error as Error).message).not.toContain('internal-auth');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('logs which branch failed and the code path, never the error text', async () => {
    const authHeaders = jest.fn(async () => {
      throw new SyntaxError(
        'Unexpected token \'M\', ..."d":MIIEvgIBAD"... is not valid JSON'
      );
    });
    const tool = createCodeExecutionTool({ authHeaders });

    const error = await tool
      .invoke({ lang: 'py', code: 'print(1)' })
      .catch((caught: unknown) => caught);

    expect((error as Error).message).not.toContain('MIIEvgIBAD');
    expect(errorSpy).toHaveBeenCalledTimes(1);
    const [logged, detail] = errorSpy.mock.calls[0];
    expect(logged).toBe(
      '[CodeExecutor] auth header resolution failed; the Code API request was never sent'
    );
    expect(detail).toEqual({ type: 'SyntaxError' });
    expect(JSON.stringify(detail)).not.toContain('MIIEvgIBAD');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('classifies past an accessor trap, which it never reaches', async () => {
    const hostile = new Proxy(new Error('boom'), {
      get(): never {
        throw new Error('accessor exploded');
      },
    });
    const tool = createCodeExecutionTool({
      authHeaders: () => Promise.reject(hostile),
    });

    await tool.invoke({ lang: 'py', code: 'print(1)' }).catch(() => undefined);

    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][1]).toEqual({ type: 'Error' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('still logs when classification itself throws', async () => {
    const hostile = new Proxy(new Error('boom'), {
      getPrototypeOf(): never {
        throw new Error('prototype exploded');
      },
    });
    const tool = createCodeExecutionTool({
      authHeaders: () => Promise.reject(hostile),
    });

    await tool.invoke({ lang: 'py', code: 'print(1)' }).catch(() => undefined);

    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][1]).toEqual({ type: 'UndescribableError' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('forwards Authorization for direct code execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const tool = createCodeExecutionTool({
      authHeaders: async () => ({ Authorization: 'Bearer code-token' }),
    });

    await tool.invoke({ lang: 'py', code: 'print(1)' });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer code-token',
        }),
      })
    );
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)
    ).not.toHaveProperty('authHeaders');
  });

  it('surfaces artifact delivery failures from direct code execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'completed',
        session_id: 'session_123',
        stdout: 'code completed\n',
        files: [],
        artifact_delivery: {
          code: 'artifact_delivery_failed',
          status: 'failed',
          attempted: 1,
          delivered: 0,
          failed: 1,
        },
      })
    );
    const tool = createCodeExecutionTool();

    const output = await tool.invoke({ lang: 'py', code: 'print(1)' });

    expect(output).toContain('stdout:\ncode completed');
    expect(output).toContain('Artifact delivery warning:');
    expect(output).toContain('do not rerun automatically');
  });

  it('warns before the file summary and echoes a normalized truncation marker', async () => {
    const files = [{ id: 'file-1', name: 'report_001.csv' }];
    const marker = {
      code: 'artifact_truncated',
      reasons: { max_files: 70 },
      skipped: ['report_070.csv'],
      skipped_count: 70,
    };
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        session_id: 'session_123',
        stdout: 'code completed\n',
        files,
        artifact_truncation: { ...marker, detail: 'ignored by normalization' },
      })
    );

    const tool = createCodeExecutionTool();
    const result = (await tool.invoke({
      name: tool.name,
      args: { lang: 'py', code: 'print(1)' },
      id: 'call-truncated',
      type: 'tool_call',
    } as never)) as { content: string; artifact?: t.CodeExecutionArtifact };

    expect(result.content).toContain('70 file(s) were omitted from delivery');
    expect(result.content).toContain('report_070.csv (1 of 70 shown)');
    expect(result.content).toContain('do not rerun automatically');
    expect(result.content.indexOf('file(s) were omitted')).toBeLessThan(
      result.content.indexOf('Generated files:')
    );
    expect(result.artifact?.files).toEqual(files);
    expect(result.artifact?.artifact_truncation).toEqual(marker);
  });

  it('warns and echoes truncation metadata even when no files were delivered', async () => {
    const marker = {
      code: 'artifact_truncated',
      reasons: { size: 1 },
      skipped: ['oversized.csv'],
      skipped_count: 1,
    };
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        session_id: 'session_123',
        stdout: 'done\n',
        files: [],
        artifact_truncation: marker,
      })
    );

    const tool = createCodeExecutionTool();
    const result = (await tool.invoke({
      name: tool.name,
      args: { lang: 'py', code: 'print(1)' },
      id: 'call-truncated-empty',
      type: 'tool_call',
    } as never)) as { content: string; artifact?: t.CodeExecutionArtifact };

    expect(result.content).toContain(
      '1 file(s) were omitted from delivery (size: 1)'
    );
    expect(result.content).not.toContain('Generated files:');
    expect(result.artifact?.artifact_truncation).toEqual(marker);
  });

  it.each([
    { reasons: { unexpected: 1 }, skipped: ['file.csv'] },
    {
      reasons: { max_files: 21 },
      skipped: Array.from({ length: 21 }, (_, index) => `file_${index}.csv`),
    },
  ])(
    'ignores malformed truncation markers in direct execution',
    async (fields) => {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({
          session_id: 'session_123',
          stdout: 'done\n',
          files: [{ id: 'file-1', name: 'kept.csv' }],
          artifact_truncation: {
            code: 'artifact_truncated',
            skipped_count: 21,
            ...fields,
          },
        })
      );

      const tool = createCodeExecutionTool();
      const result = (await tool.invoke({
        name: tool.name,
        args: { lang: 'py', code: 'print(1)' },
        id: 'call-malformed-truncation',
        type: 'tool_call',
      } as never)) as { content: string; artifact?: t.CodeExecutionArtifact };

      expect(result.content).not.toContain('omitted from delivery');
      expect(result.artifact?.artifact_truncation).toBeUndefined();
      expect(result.artifact?.files).toHaveLength(1);
    }
  );

  it.each([
    { files: [{ id: 'file-1', name: 'kept.csv' }] },
    { files: [] },
  ])('surfaces Bash truncation with files $files', async ({ files }) => {
    const marker = {
      code: 'artifact_truncated',
      reasons: { max_files: 2 },
      skipped: ['omitted.csv'],
      skipped_count: 2,
    };
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        session_id: 'session_123',
        stdout: 'bash completed\n',
        files,
        artifact_truncation: { ...marker, detail: 'not trusted' },
      })
    );
    const tool = createBashExecutionTool();

    const result = (await tool.invoke({
      name: tool.name,
      args: { command: 'echo done' },
      id: 'call-bash-truncated',
      type: 'tool_call',
    } as never)) as { content: string; artifact?: t.CodeExecutionArtifact };

    expect(result.content).toContain('2 file(s) were omitted from delivery');
    expect(result.content).toContain('omitted.csv (1 of 2 shown)');
    expect(result.content).toContain('do not rerun automatically');
    expect(result.content.includes('Generated files:')).toBe(files.length > 0);
    if (files.length > 0) {
      expect(result.content.indexOf('file(s) were omitted')).toBeLessThan(
        result.content.indexOf('Generated files:')
      );
    }
    expect(result.artifact?.artifact_truncation).toEqual(marker);
    expect(result.artifact?.files).toEqual(files.length > 0 ? files : undefined);
  });

  it('ignores invalid truncation metadata in direct Bash execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        session_id: 'session_123',
        stdout: 'done\n',
        files: [],
        artifact_truncation: {
          code: 'artifact_truncated',
          reasons: { unexpected: 1 },
          skipped: ['omitted.csv'],
          skipped_count: 1,
        },
      })
    );
    const tool = createBashExecutionTool();

    const result = (await tool.invoke({
      name: tool.name,
      args: { command: 'echo done' },
      id: 'call-bash-malformed',
      type: 'tool_call',
    } as never)) as { content: string; artifact?: t.CodeExecutionArtifact };

    expect(result.content).not.toContain('omitted from delivery');
    expect(result.artifact?.artifact_truncation).toBeUndefined();
  });

  it('surfaces artifact delivery failures from direct bash execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        session_id: 'session_123',
        stdout: 'bash completed\n',
        files: [],
        artifact_delivery: {
          code: 'artifact_delivery_failed',
          status: 'failed',
          attempted: 1,
          delivered: 0,
          failed: 1,
        },
      })
    );
    const tool = createBashExecutionTool();

    const output = await tool.invoke({ command: 'echo done' });

    expect(output).toContain('stdout:\nbash completed');
    expect(output).toContain('Artifact delivery warning:');
    expect(output).toContain('do not rerun automatically');
  });

  it.each([
    ['code', createCodeExecutionTool, { lang: 'py', code: 'print(1)' }],
    ['bash', createBashExecutionTool, { command: 'rm data.csv' }],
  ] as const)(
    'preserves deleted file paths in direct %s execution artifacts',
    async (_kind, createTool, args) => {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({
          session_id: 'session_123',
          stdout: 'done\n',
          files: [],
          deleted_files: ['data.csv'],
        })
      );
      const tool = createTool();

      const result = await tool.invoke({
        name: tool.name,
        args,
        id: 'call-delete',
        type: 'tool_call',
      } as never) as { artifact?: t.CodeExecutionArtifact };

      expect(result.artifact?.deleted_files).toEqual(['data.csv']);
    }
  );

  it('routes direct code tools by trusted per-agent profile', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const defaultTool = createCodeExecutionTool({
      baseUrl: 'https://code-default.example.com',
      executionProfile: 'default',
      statefulSessions: false,
    });
    const statefulTool = createCodeExecutionTool({
      baseUrl: 'https://code-stateful.example.com',
      executionProfile: 'stateful',
      runtimeSessionHint: 'user-123',
      statefulSessions: true,
    });

    await defaultTool.invoke({ lang: 'py', code: 'print(1)' }, {
      toolCall: { _runtime_session_hint: 'graph-wide-hint' },
    } as unknown as RunnableConfig);
    await statefulTool.invoke({ lang: 'py', code: 'print(1)' }, {
      toolCall: { _runtime_session_hint: 'wrong-agent-hint' },
    } as unknown as RunnableConfig);

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code-default.example.com/exec'
    );
    expect(requestHeadersAt(0)).toEqual(
      expect.objectContaining({ 'X-CodeAPI-Expected-Profile': 'default' })
    );
    expect(requestBodyAt(0)).not.toHaveProperty('runtime_session_hint');
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      'https://code-stateful.example.com/exec'
    );
    expect(requestHeadersAt(1)).toEqual(
      expect.objectContaining({ 'X-CodeAPI-Expected-Profile': 'stateful' })
    );
    expect(requestBodyAt(1).runtime_session_hint).toBe('user-123');
  });

  it('normalizes profile URLs and falls back from empty factory hints', async () => {
    const codeTool = createCodeExecutionTool({
      baseUrl: 'https://code-stateful.example.com/',
      executionProfile: 'stateful',
      runtimeSessionHint: '',
      statefulSessions: true,
    });
    const bashTool = createBashExecutionTool({
      baseUrl: 'https://code-stateful.example.com///',
      executionProfile: 'stateful',
      runtimeSessionHint: '',
      statefulSessions: true,
    });

    await codeTool.invoke({ lang: 'py', code: 'print(1)' }, {
      toolCall: { _runtime_session_hint: 'thread-fallback' },
    } as unknown as RunnableConfig);
    await bashTool.invoke({ command: 'echo 1' }, {
      toolCall: { _runtime_session_hint: 'thread-fallback' },
    } as unknown as RunnableConfig);

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code-stateful.example.com/exec'
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      'https://code-stateful.example.com/exec'
    );
    expect(requestBodyAt(0).runtime_session_hint).toBe('thread-fallback');
    expect(requestBodyAt(1).runtime_session_hint).toBe('thread-fallback');
  });

  it('tolerates null params for direct code execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const tool = createCodeExecutionTool(null);

    await expect(
      tool.invoke({ lang: 'py', code: 'print(1)' })
    ).resolves.toBeDefined();
  });

  it('forwards Authorization for bash execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const tool = createBashExecutionTool({
      authHeaders: { Authorization: 'Bearer bash-token' },
    });

    await tool.invoke({ command: 'echo 1' });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer bash-token',
        }),
      })
    );
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)
    ).not.toHaveProperty('authHeaders');
  });

  it('executes injected skill files inside a selected attached workspace', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'completed',
        session_id: 'session_123',
        stdout: 'skill output\n',
        files: [
          {
            id: 'artifact-1',
            name: 'report.txt',
            storage_session_id: 'session_123',
          },
        ],
      })
    );
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
      authHeaders: { Authorization: 'Bearer worker-token' },
      workspaceId: 'project:a',
    });

    const output = await tool.invoke(
      { command: 'node "$LIBRECHAT_CODE_DATA_DIR/skills/report/run.js"' },
      {
        toolCall: {
          _injected_files: [
            {
              id: 'file-1',
              resource_id: 'skill-1',
              storage_session_id: 'storage-1',
              name: 'skills/report/run.js',
              kind: 'skill',
              version: 1,
            },
          ],
        },
      } as unknown as RunnableConfig
    );

    expect(output).toContain('Execution artifacts: 1 file(s)');
    expect(output).not.toContain(
      'persisted file(s) are available in /mnt/data'
    );
    expect(tool.description).toContain('selected persistent project');

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code.example.com/v1/exec/programmatic'
    );
    expect(requestHeadersAt(0)).toEqual(
      expect.objectContaining({
        Authorization: 'Bearer worker-token',
        'X-LibreChat-Code-Workspace-ID': 'project:a',
      })
    );
    expect(requestBodyAt(0)).toEqual(
      expect.objectContaining({
        lang: 'bash',
        tools: [],
        files: [expect.objectContaining({ name: 'skills/report/run.js' })],
      })
    );
    expect(String(requestBodyAt(0).code)).toContain(': &\nwait "$!"');
  });

  it('binds direct attached Bash to a trusted workspace instance', async () => {
    fetchMock.mockResolvedValueOnce(completedResponse('done'));
    const workspaceInstanceId = 'b'.repeat(64);
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
      workspaceId: 'project-a',
      workspaceInstanceId,
    });

    await tool.invoke({ command: 'pwd' });

    expect(requestBodyAt(0)).toMatchObject({
      workspace_instance_id: workspaceInstanceId,
    });
  });

  it('does not let raw Bash arguments override a trusted workspace instance', async () => {
    fetchMock.mockResolvedValueOnce(completedResponse('done'));
    const workspaceInstanceId = 'b'.repeat(64);
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
      workspaceId: 'project-a',
      workspaceInstanceId,
    });

    await tool.invoke({
      command: 'pwd',
      workspace_instance_id: 'a'.repeat(64),
    });

    expect(requestBodyAt(0)).toMatchObject({
      workspace_instance_id: workspaceInstanceId,
    });
  });

  it('drops raw workspace instances when direct Bash has no trusted binding', async () => {
    fetchMock.mockResolvedValueOnce(completedResponse('done'));
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
    });

    await tool.invoke({
      command: 'pwd',
      workspace_instance_id: 'a'.repeat(64),
    });

    expect(requestBodyAt(0)).not.toHaveProperty('workspace_instance_id');
  });

  it('rejects malformed or unscoped direct Bash workspace instances', () => {
    expect(() =>
      createBashExecutionTool({
        workspaceId: 'project-a',
        workspaceInstanceId: 'not-a-sha256',
      })
    ).toThrow('Invalid attached workspace instance identifier');
    expect(() =>
      createBashExecutionTool({ workspaceInstanceId: 'b'.repeat(64) })
    ).toThrow('Invalid attached workspace instance identifier');
  });

  it.each(['', '   '])(
    'treats a blank direct Bash workspace instance as absent: %j',
    async (workspaceInstanceId) => {
      fetchMock.mockResolvedValueOnce(completedResponse('done'));
      const tool = createBashExecutionTool({
        baseUrl: 'https://code.example.com/v1',
        workspaceId: 'project-a',
        workspaceInstanceId,
      });

      await tool.invoke({ command: 'pwd' });

      expect(requestBodyAt(0)).not.toHaveProperty('workspace_instance_id');
    }
  );

  it('surfaces a selected-workspace execution error instead of formatting success', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error: 'sandbox command failed',
        session_id: 'session_123',
      })
    );
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
      workspaceId: 'project:a',
    });

    await expect(tool.invoke({ command: 'exit 1' })).rejects.toThrow(
      'Code execution failed.'
    );
  });

  it('preserves args and sends explicit cancellation for selected-workspace Bash', async () => {
    const controller = new AbortController();
    fetchMock
      .mockImplementationOnce((_url, rawInit) => {
        const init = rawInit as RequestInit;
        return new Promise((_resolve, reject) => {
          init.signal?.addEventListener(
            'abort',
            () => reject(new DOMException('aborted', 'AbortError')),
            { once: true }
          );
        });
      })
      .mockResolvedValueOnce(jsonResponse({ status: 'cancellation_requested' }));
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
      authHeaders: { Authorization: 'Bearer worker-token' },
      executionProfile: 'stateful',
      workspaceId: 'project:a',
    });

    const request = tool
      .invoke(
        { command: 'printf "%s" "$1"', args: ['value with spaces'] },
        { signal: controller.signal } as unknown as RunnableConfig
      )
      .catch((error: unknown) => error);
    await new Promise<void>((resolve) => setImmediate(resolve));
    const requestId = requestHeadersAt(0)['X-LibreChat-Code-Request-ID'];
    expect(String(requestBodyAt(0).code)).toContain(
      'bash -c \'printf "%s" "$1"\' -- \'value with spaces\''
    );
    expect(requestBodyAt(0)).not.toHaveProperty('args');
    controller.abort();
    await request;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      'https://code.example.com/v1/exec/programmatic/cancel'
    );
    expect(requestHeadersAt(1)).toMatchObject({
      Authorization: 'Bearer worker-token',
      'X-CodeAPI-Expected-Profile': 'stateful',
    });
    expect(requestBodyAt(1)).toEqual({ request_id: requestId });
  });

  it('points attached /tmp reminders at the selected project', async () => {
    fetchMock.mockResolvedValueOnce(completedResponse('done'));
    const tool = createBashExecutionTool({
      baseUrl: 'https://code.example.com/v1',
      workspaceId: 'project:a',
    });

    const output = await tool.invoke({ command: 'touch /tmp/transient' });

    expect(output).toContain('write files needed later into the selected project');
    expect(output).not.toContain('use /mnt/data for files needed later');
  });

  it('routes bash tools by trusted per-agent profile', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const tool = createBashExecutionTool({
      baseUrl: 'https://code-stateful.example.com',
      executionProfile: 'stateful',
      runtimeSessionHint: 'user-123',
      statefulSessions: true,
    });

    await tool.invoke({ command: 'echo 1' }, {
      toolCall: { _runtime_session_hint: 'wrong-agent-hint' },
    } as unknown as RunnableConfig);

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code-stateful.example.com/exec'
    );
    expect(requestHeadersAt(0)).toEqual(
      expect.objectContaining({ 'X-CodeAPI-Expected-Profile': 'stateful' })
    );
    expect(requestBodyAt(0).runtime_session_hint).toBe('user-123');
  });

  it('redacts the CodeAPI endpoint and response body on direct execution failures', async () => {
    fetchMock.mockResolvedValueOnce(errorResponse(404, 'Cannot POST /exec'));
    const tool = createBashExecutionTool();

    const error = await tool
      .invoke({ command: 'echo 1' })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution is temporarily unavailable. Please retry.'
    );
    expect((error as Error).message).not.toContain('/exec');
    expect((error as Error).message).not.toContain('Cannot POST');
  });

  it.each([400, 409, 422, 500, 503])(
    'classifies worker capability mismatches independently of HTTP %s',
    async (status) => {
      fetchMock.mockResolvedValueOnce(
        errorResponse(
          status,
          JSON.stringify({
            error: 'bridge_worker_mismatch',
            message:
              'Code environment does not support this execution; select a compatible worker',
          })
        )
      );
      const error = await makeRequest('https://code.example/exec', {}).catch(
        (caught: unknown) => caught
      );
      expect(error).toBeInstanceOf(CodeApiRequestError);
      expect(error).toHaveProperty('retryable', false);
      expect((error as Error).message).toContain('stateful workspace');
      expect((error as Error).message).toContain('Do not retry');
      expect((error as Error).message).not.toContain('temporarily');
    }
  );

  it.each([
    'BRIDGE_WORKER_MISMATCH',
    'execution_profile_mismatch',
    'capability_mismatch',
    'unsupported_capability',
  ])(
    'recognizes structured capability code %s without exposing upstream text',
    async (code) => {
      fetchMock.mockResolvedValueOnce(
        errorResponse(
          503,
          JSON.stringify({
            code,
            message: 'Bearer secret-token from private.internal',
          })
        )
      );
      const error = await makeRequest('https://code.example/exec', {}).catch(
        (caught: unknown) => caught
      );
      expect(error).toHaveProperty('retryable', false);
      expect((error as Error).message).not.toContain('secret-token');
      expect((error as Error).message).not.toContain('private.internal');
    }
  );

  it.each([
    ['Bash', () => createBashExecutionTool().invoke({ command: 'echo 1' })],
    [
      'Code',
      () => createCodeExecutionTool().invoke({ lang: 'py', code: 'print(1)' }),
    ],
    [
      'Python PTC',
      () =>
        createProgrammaticToolCallingTool().invoke({ code: 'print(1)' }, {
          toolCall: { toolMap: toolMap(), toolDefs },
        } as RunnableConfig),
    ],
  ] as const)(
    'preserves non-retryable errors through the %s wrapper',
    async (_name, invoke) => {
      fetchMock.mockResolvedValueOnce(
        errorResponse(409, JSON.stringify({ error: 'bridge_worker_mismatch' }))
      );
      const error = await invoke().catch((caught: unknown) => caught);
      expect(error).toBeInstanceOf(CodeApiRequestError);
      expect(error).toHaveProperty('retryable', false);
    }
  );

  it('does not tell programmatic Bash to rerun files after a permanent mismatch', async () => {
    fetchMock.mockResolvedValueOnce(
      errorResponse(
        409,
        JSON.stringify({
          error: 'BRIDGE_WORKER_MISMATCH',
          message:
            'Bridge worker personal-vm does not provide a stateful workspace',
        })
      )
    );
    const error = await createBashProgrammaticToolCallingTool()
      .invoke({ code: 'echo result > /mnt/data/result.txt' }, {
        toolCall: { toolMap: toolMap(), toolDefs },
      } as RunnableConfig)
      .catch((caught: unknown) => caught);
    expect(error).toHaveProperty('retryable', false);
    expect((error as Error).message).toContain(
      'does not provide a stateful workspace'
    );
    expect((error as Error).message).not.toContain('personal-vm');
    expect((error as Error).message).not.toContain('fix the error and rerun');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    'not JSON',
    'null',
    '{"error":"bridge_worker_offline"}',
    '{"error":"bridge_worker_busy"}',
  ])('keeps unavailable failures retryable for %s', async (body) => {
    fetchMock.mockResolvedValueOnce(errorResponse(503, body));
    const error = await makeRequest('https://code.example/exec', {}).catch(
      (caught: unknown) => caught
    );
    expect(error).toHaveProperty('retryable', true);
    expect((error as Error).message).toContain('temporarily unavailable');
  });

  it.each([401, 403])(
    'reports CodeAPI HTTP %s authorization failures as non-retryable',
    async (status) => {
      fetchMock.mockResolvedValueOnce(
        errorResponse(
          status,
          'Invalid bearer token for codeapi.internal.svc.cluster.local'
        )
      );
      const tool = createBashExecutionTool();

      const error = await tool
        .invoke({ command: 'echo 1' })
        .catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toContain(
        'Code execution is not authorized. Verify access before trying again.'
      );
      expect((error as Error).message).not.toContain('Please retry');
      expect((error as Error).message).not.toContain('svc.cluster.local');
      expect((error as Error).message).not.toContain('Invalid bearer token');
    }
  );

  it.each([401, 403])(
    'logs the status and authority behind the %s authorization error, never the body',
    async (status) => {
      fetchMock.mockResolvedValueOnce(
        errorResponse(
          status,
          '{"received":"Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1In0.c2ln"}'
        )
      );
      const tool = createBashExecutionTool({
        baseUrl: 'https://gateway.example/v1',
        executionProfile: 'stateful',
      });

      await tool.invoke({ command: 'echo 1' }).catch(() => undefined);

      expect(errorSpy).toHaveBeenCalledTimes(1);
      const [logged, detail] = errorSpy.mock.calls[0];
      expect(logged).toBe('[CodeExecutor] Code API rejected the request');
      expect(detail).toEqual({
        method: 'POST',
        profile: 'stateful',
        status,
      });
      expect(JSON.stringify(detail)).not.toContain('eyJhbGciOiJSUzI1NiJ9');
    }
  );

  it('names the backend by profile, never by its configured address', async () => {
    fetchMock.mockResolvedValueOnce(errorResponse(500, 'upstream exploded'));
    const tool = createBashExecutionTool({
      baseUrl: 'https://MIIEvgIBAD.gateway.example/t/MIIEvgIBAD/v1',
    });

    await tool.invoke({ command: 'echo 1' }).catch(() => undefined);

    expect(errorSpy.mock.calls[0][1]).toEqual({
      method: 'POST',
      profile: 'unset',
      status: 500,
    });
    expect(JSON.stringify(errorSpy.mock.calls[0][1])).not.toContain(
      'MIIEvgIBAD'
    );
  });

  it('classifies a failed session file lookup instead of quoting it', async () => {
    const files = await fetchSessionFiles(
      'https://gateway.example/v1',
      'session_123',
      undefined,
      () => {
        throw new SyntaxError(
          'bad JWK ..."d":MIIEvgIBAD"... is not valid JSON'
        );
      }
    );

    expect(files).toEqual([]);
    /* The resolver has already normalized the callback's SyntaxError into a
       CodeApiRequestError by this point; the classification is deliberately
       whatever reached the catch, never text quoted from it. */
    expect(warnSpy).toHaveBeenCalledWith(
      '[ProgrammaticToolCalling] session file lookup failed; continuing without input files',
      { type: 'Error' }
    );
    expect(JSON.stringify(warnSpy.mock.calls)).not.toContain('MIIEvgIBAD');
    expect(JSON.stringify(warnSpy.mock.calls)).not.toContain('session_123');
  });

  it('carries no host text into any diagnostic, across every failure mode', async () => {
    const secret = 'MIIEvgIBADANBgkqhkiG9w0BAQEF';
    const debugSpy = jest
      .spyOn(console, 'debug')
      .mockImplementation(() => undefined);

    const forged = new Error(`parse failed\n    at ${secret}`);
    forged.name = secret;
    await createCodeExecutionTool({
      authHeaders: () => Promise.reject(forged),
    })
      .invoke({ lang: 'py', code: 'print(1)' })
      .catch(() => undefined);

    fetchMock.mockResolvedValueOnce(
      errorResponse(401, `{"received":"Bearer ${secret}"}`)
    );
    await createBashExecutionTool({
      baseUrl: `https://${secret}.gateway.example/t/${secret}/v1`,
    })
      .invoke({ command: 'echo 1' })
      .catch(() => undefined);

    await createCodeExecutionTool({ session_id: secret })
      .invoke({ lang: 'py', code: 'print(1)' }, {
        toolCall: { session_id: secret },
      } as unknown as RunnableConfig)
      .catch(() => undefined);

    const logged = JSON.stringify([
      errorSpy.mock.calls,
      warnSpy.mock.calls,
      debugSpy.mock.calls,
    ]);
    expect(logged).not.toContain(secret);
    expect(logged).toContain('[CodeExecutor]');
    debugSpy.mockRestore();
  });

  it('routes the bash programmatic executor through the same guard', async () => {
    const debugSpy = jest
      .spyOn(console, 'debug')
      .mockImplementation(() => undefined);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: 'completed', session_id: 's', stdout: 'ok' })
    );

    await createBashProgrammaticToolCallingTool()
      .invoke({ code: 'lookup_user "{}"' }, {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
          session_id: 'MIIEvgIBADANBgkqhkiG9w0BAQEF',
        },
      } as unknown as RunnableConfig)
      .catch(() => undefined);

    expect(debugSpy).toHaveBeenCalledWith(
      '[BashProgrammaticToolCalling] session carried no injected files; exec will run without input files',
      { files: 'none' }
    );
    expect(JSON.stringify(debugSpy.mock.calls)).not.toContain('MIIEvgIBAD');
    debugSpy.mockRestore();
  });

  it('releases error response bodies after bounded classification', async () => {
    const destroy = jest.fn();
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 503,
      body: { destroy },
      text: jest.fn(async () => ''),
    });
    await createBashExecutionTool()
      .invoke({ command: 'echo 1' })
      .catch(() => undefined);

    expect(destroy).toHaveBeenCalledTimes(1);

    const rateLimitedDestroy = jest.fn();
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 429,
      body: { destroy: rateLimitedDestroy },
      text: jest.fn(async () =>
        JSON.stringify({ error: 'rate_limited', retry_after_seconds: 3 })
      ),
    });
    const error = await createBashExecutionTool()
      .invoke({ command: 'echo 1' })
      .catch((caught: unknown) => caught);

    expect(rateLimitedDestroy).toHaveBeenCalledTimes(1);
    expect((error as Error).message).toContain('Retry after 3 seconds');
  });

  it('classifies a real streamed error envelope', async () => {
    const body = Readable.from([
      Buffer.from('{"error":"bridge_'),
      Buffer.from('worker_mismatch"}'),
    ]);
    const message = await buildCodeApiHttpErrorMessage('POST', '', {
      status: 409,
      body,
      text: async () => '',
    });
    expect(message).toContain('permanent capability mismatch');
    expect(body.destroyed).toBe(true);
  });

  it('stops reading oversized error bodies', async () => {
    const body = Readable.from([
      Buffer.alloc(64 * 1024 + 1),
      Buffer.from('{"error":"bridge_worker_mismatch"}'),
    ]);
    const message = await buildCodeApiHttpErrorMessage('POST', '', {
      status: 503,
      body,
      text: async () => '',
    });
    expect(message).toContain('temporarily unavailable');
    expect(body.destroyed).toBe(true);
  });

  it('bounds stalled error reads and releases the stream', async () => {
    jest.useFakeTimers();
    const body = new Readable({ read() {} });
    try {
      const pending = buildCodeApiHttpErrorMessage('POST', '', {
        status: 503,
        body,
        text: async () => '',
      });
      await jest.advanceTimersByTimeAsync(1_000);
      expect(await pending).toContain('temporarily unavailable');
      expect(body.destroyed).toBe(true);
    } finally {
      jest.useRealTimers();
    }
  });

  it('logs the rejection before the response body is drained', async () => {
    let releaseBody: (value: string) => void = () => undefined;
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: jest.fn(
        () =>
          new Promise<string>((resolve) => {
            releaseBody = resolve;
          })
      ),
    });
    const tool = createBashExecutionTool({ executionProfile: 'default' });

    const pending = tool.invoke({ command: 'echo 1' }).catch(() => undefined);
    await new Promise((resolve) => setImmediate(resolve));

    expect(errorSpy).toHaveBeenCalledWith(
      '[CodeExecutor] Code API rejected the request',
      { method: 'POST', profile: 'default', status: 503 }
    );

    releaseBody('');
    await pending;
  });

  it.each([
    [
      'a writable name',
      () => Object.assign(new Error('boom'), { name: 'MIIEvgIBAD' }),
    ],
    [
      'a message that forges a stack frame',
      () => new Error('failed\n    at MIIEvgIBAD'),
    ],
  ])('logs no host text from %s', async (_label, build) => {
    const tool = createCodeExecutionTool({
      authHeaders: () => Promise.reject(build()),
    });

    await tool.invoke({ lang: 'py', code: 'print(1)' }).catch(() => undefined);

    expect(errorSpy.mock.calls[0][1]).toEqual({ type: 'Error' });
    expect(JSON.stringify(errorSpy.mock.calls[0][1])).not.toContain(
      'MIIEvgIBAD'
    );
  });

  it('leaves a recoverable file lookup to report an auth failure once', async () => {
    const files = await fetchSessionFiles(
      'https://gateway.example/v1',
      'session_123',
      undefined,
      () => {
        throw new Error('signing key is not configured');
      }
    );

    expect(files).toEqual([]);
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('leaves a recoverable file lookup to report its own outcome', async () => {
    fetchMock.mockResolvedValueOnce(errorResponse(404, 'no such session'));

    const files = await fetchSessionFiles(
      'https://gateway.example/v1',
      'session_123'
    );

    expect(files).toEqual([]);
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it('logs rate-limited rejections at warn so retries do not read as failures', async () => {
    fetchMock.mockResolvedValueOnce(
      errorResponse(429, JSON.stringify({ error: 'rate_limited' }))
    );
    const tool = createBashExecutionTool();

    await tool.invoke({ command: 'echo 1' }).catch(() => undefined);

    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(
      '[CodeExecutor] Code API rejected the request'
    );
  });

  it('preserves only a bounded retry delay from CodeAPI rate-limit failures', async () => {
    fetchMock.mockResolvedValueOnce(
      errorResponse(
        429,
        JSON.stringify({
          error: 'rate_limited',
          message:
            'Too many CodeAPI execution requests from internal deployment details.',
          retry_after_seconds: 8.2,
        })
      )
    );
    const tool = createBashExecutionTool();

    const error = await tool
      .invoke({ command: 'echo 1' })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution is temporarily rate-limited. Retry after 9 seconds.'
    );
    expect((error as Error).message).not.toContain('CodeAPI');
    expect((error as Error).message).not.toContain('internal deployment');
  });

  it('redacts network details from direct execution failures', async () => {
    fetchMock.mockRejectedValueOnce(
      new Error(
        'request to http://codeapi.internal.svc.cluster.local/exec failed: getaddrinfo ENOTFOUND'
      )
    );
    const tool = createBashExecutionTool();

    const error = await tool
      .invoke({ command: 'echo 1' })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution is temporarily unavailable. Please retry.'
    );
    expect((error as Error).message).not.toContain('svc.cluster.local');
    expect((error as Error).message).not.toContain('ENOTFOUND');
  });

  it('redacts network details from programmatic execution failures', async () => {
    fetchMock.mockRejectedValueOnce(
      new Error(
        'request to http://codeapi.internal.svc.cluster.local/exec/programmatic failed'
      )
    );
    const tool = createProgrammaticToolCallingTool();

    const error = await tool
      .invoke(
        { code: 'print("hello")' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution is temporarily unavailable. Please retry.'
    );
    expect((error as Error).message).not.toContain('svc.cluster.local');
  });

  it('redacts CodeAPI programmatic errors while preserving execution stderr', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error:
          'sandbox worker codeapi-runtime-7f9d failed in internal namespace',
        stderr: 'NameError: tenant_variable is not defined',
      })
    );
    const tool = createProgrammaticToolCallingTool();

    const error = await tool
      .invoke(
        { code: 'print(tenant_variable)' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain('Code execution failed.');
    expect((error as Error).message).toContain(
      'NameError: tenant_variable is not defined'
    );
    expect((error as Error).message).not.toContain('codeapi-runtime');
    expect((error as Error).message).not.toContain('internal namespace');
  });

  it('preserves an allowlisted programmatic execution error when stderr is absent', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error: 'Time limit exceeded',
      })
    );
    const tool = createProgrammaticToolCallingTool();

    const error = await tool
      .invoke(
        { code: 'while True: pass' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution failed. Execution exceeded the time limit.'
    );
  });

  it('preserves an allowlisted execution error alongside stderr', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error: 'Time limit exceeded',
        stderr: 'processed 42 records before termination',
      })
    );
    const tool = createProgrammaticToolCallingTool();

    const error = await tool
      .invoke(
        { code: 'while True: process_next_record()' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution failed. Execution exceeded the time limit.'
    );
    expect((error as Error).message).toContain(
      'Stderr:\nprocessed 42 records before termination'
    );
  });

  it('keeps arbitrary programmatic execution errors redacted when stderr is absent', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error:
          'sandbox worker codeapi-runtime-7f9d failed in internal namespace',
      })
    );
    const tool = createProgrammaticToolCallingTool();

    const error = await tool
      .invoke(
        { code: 'print("hello")' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain('Code execution failed.');
    expect((error as Error).message).not.toContain('codeapi-runtime');
    expect((error as Error).message).not.toContain('internal namespace');
  });

  it('preserves an allowlisted bash execution error when stderr is absent', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error: 'Out of memory',
      })
    );
    const tool = createBashProgrammaticToolCallingTool();

    const error = await tool
      .invoke(
        { code: 'lookup_user "{}"' },
        {
          toolCall: {
            name: 'bash_programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      'Code execution failed. Execution exceeded the memory limit.'
    );
  });

  it('forwards Authorization on programmatic initial and continuation requests', async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          status: 'tool_call_required',
          continuation_token: 'continue_123',
          tool_calls: [{ id: 'call_1', name: 'lookup_user', input: {} }],
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          status: 'completed',
          session_id: 'session_123',
          stdout: 'done',
          files: [
            {
              id: 'artifact-1',
              name: 'result.txt',
              storage_session_id: 'session_123',
            },
          ],
        })
      );

    const tool = createProgrammaticToolCallingTool({
      authHeaders: () => ({ Authorization: 'Bearer ptc-token' }),
      executionProfile: 'stateful',
      runtimeSessionHint: 'user-123',
    });

    await tool.invoke(
      { code: 'result = await lookup_user()\nprint(result)' },
      {
        toolCall: {
          name: 'programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const call of fetchMock.mock.calls) {
      expect(call[1]).toEqual(
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer ptc-token',
            'X-CodeAPI-Expected-Profile': 'stateful',
          }),
        })
      );
    }
    expect(requestBodyAt(0).runtime_session_hint).toBe('user-123');
    expect(requestBodyAt(1)).not.toHaveProperty('runtime_session_hint');
  });

  it('binds bash programmatic initial and continuation requests to the selected workspace', async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          status: 'tool_call_required',
          continuation_token: 'continue_workspace',
          tool_calls: [{ id: 'call_1', name: 'lookup_user', input: {} }],
        })
      )
      .mockResolvedValueOnce(completedResponse('done'));
    const tool = createBashProgrammaticToolCallingTool({
      baseUrl: 'https://code.example.com',
      workspaceId: 'project-a',
      authHeaders: { Authorization: 'Bearer workspace-token' },
    });

    expect(tool.description).toContain('selected persistent workspace');
    expect(tool.description).not.toContain('CRITICAL - STATELESS EXECUTION');
    expect(
      (tool.schema as { properties: { code: { description: string } } })
        .properties.code.description
    ).toContain('selected persistent workspace');

    await tool.invoke(
      { code: 'lookup_user "{}"' },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (let i = 0; i < 2; i++) {
      expect(requestHeadersAt(i)).toMatchObject({
        Authorization: 'Bearer workspace-token',
        'X-LibreChat-Code-Workspace-ID': 'project-a',
      });
    }
  });

  it('binds the initial programmatic request to a trusted workspace instance', async () => {
    fetchMock.mockResolvedValueOnce(completedResponse('done'));
    const workspaceInstanceId = 'a'.repeat(64);
    const tool = createBashProgrammaticToolCallingTool({
      baseUrl: 'https://code.example.com',
      workspaceId: 'project-a',
      workspaceInstanceId,
    });

    await tool.invoke(
      { code: 'pwd', tool_manifest: [] },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: new Map(),
          toolDefs: [],
        },
      }
    );

    expect(requestBodyAt(0)).toMatchObject({
      workspace_instance_id: workspaceInstanceId,
    });
  });

  it('rejects malformed or unscoped workspace instance identifiers', () => {
    expect(() =>
      createBashProgrammaticToolCallingTool({
        workspaceId: 'project-a',
        workspaceInstanceId: 'not-a-sha256',
      })
    ).toThrow('Invalid attached workspace instance identifier');
    expect(() =>
      createBashProgrammaticToolCallingTool({
        workspaceInstanceId: 'a'.repeat(64),
      })
    ).toThrow('Invalid attached workspace instance identifier');
  });

  it.each(['', '   '])(
    'treats a blank programmatic workspace instance as absent: %j',
    async (workspaceInstanceId) => {
      fetchMock.mockResolvedValueOnce(completedResponse('done'));
      const tool = createBashProgrammaticToolCallingTool({
        baseUrl: 'https://code.example.com',
        workspaceId: 'project-a',
        workspaceInstanceId,
      });

      await tool.invoke(
        { code: 'pwd', tool_manifest: [] },
        {
          toolCall: {
            name: 'bash_programmatic_code_execution',
            args: {},
            toolMap: new Map(),
            toolDefs: [],
          },
        }
      );

      expect(requestBodyAt(0)).not.toHaveProperty('workspace_instance_id');
    }
  );

  it('keeps tool-free attached bash execution on the programmatic workspace path', async () => {
    fetchMock.mockResolvedValueOnce(completedResponse('done'));
    const tool = createBashProgrammaticToolCallingTool({
      baseUrl: 'https://code.example.com',
      workspaceId: 'project-a',
    });

    await tool.invoke(
      { code: 'pwd', tool_manifest: [] },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: new Map(),
          toolDefs: [],
        },
      }
    );

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code.example.com/exec/programmatic'
    );
    expect(requestHeadersAt(0)).toMatchObject({
      'X-LibreChat-Code-Workspace-ID': 'project-a',
    });
  });

  it('keeps explicit default PTC stateless despite a graph-wide hint', async () => {
    const tool = createProgrammaticToolCallingTool({
      baseUrl: 'https://code-default.example.com',
      executionProfile: 'default',
    });

    await tool.invoke(
      { code: 'result = await lookup_user()\nprint(result)' },
      {
        toolCall: {
          name: 'programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
          _runtime_session_hint: 'graph-wide-hint',
        },
      }
    );

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code-default.example.com/exec/programmatic'
    );
    expect(requestHeadersAt(0)).toEqual(
      expect.objectContaining({ 'X-CodeAPI-Expected-Profile': 'default' })
    );
    expect(requestBodyAt(0)).not.toHaveProperty('runtime_session_hint');
  });

  it.each([
    ['python', createProgrammaticToolCallingTool],
    ['bash', createBashProgrammaticToolCallingTool],
  ] as const)(
    'normalizes %s PTC URLs and falls back from empty factory hints',
    async (_name, createTool) => {
      const tool = createTool({
        baseUrl: 'https://code-stateful.example.com/',
        executionProfile: 'stateful',
        runtimeSessionHint: '',
      });

      await tool.invoke(
        { code: 'print("ok")' },
        {
          toolCall: {
            name: 'programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
            _runtime_session_hint: 'thread-fallback',
          },
        }
      );

      expect(fetchMock.mock.calls[0]?.[0]).toBe(
        'https://code-stateful.example.com/exec/programmatic'
      );
      expect(requestBodyAt(0).runtime_session_hint).toBe('thread-fallback');
    }
  );

  it('defaults programmatic timeout to the configured CodeAPI run cap', async () => {
    const tool = createProgrammaticToolCallingTool({
      runTimeoutMs: 15000,
    });

    await tool.invoke(
      { code: 'result = await lookup_user()\nprint(result)' },
      {
        toolCall: {
          name: 'programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(requestBodyAt(0).timeout).toBe(15000);
  });

  it('accepts larger programmatic timeout inputs and clamps before execution', async () => {
    const tool = createProgrammaticToolCallingTool({
      runTimeoutMs: 15000,
    });

    await tool.invoke(
      { code: 'result = await lookup_user()\nprint(result)', timeout: 30000 },
      {
        toolCall: {
          name: 'programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(requestBodyAt(0).timeout).toBe(15000);
  });

  it('defaults bash programmatic timeout to the configured CodeAPI run cap', async () => {
    const tool = createBashProgrammaticToolCallingTool({
      runTimeoutMs: 15000,
    });

    await tool.invoke(
      { code: 'lookup_user "{}"' },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(requestBodyAt(0).timeout).toBe(15000);
  });

  it('accepts larger bash programmatic timeout inputs and clamps before execution', async () => {
    const tool = createBashProgrammaticToolCallingTool({
      runTimeoutMs: 15000,
    });

    await tool.invoke(
      { code: 'lookup_user "{}"', timeout: 30000 },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(requestBodyAt(0).timeout).toBe(15000);
  });

  it('initializes Bash’s last background PID before strict user code', async () => {
    const tool = createBashProgrammaticToolCallingTool();

    await tool.invoke(
      { code: 'set -euo pipefail\nlookup_user "{}"' },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(requestBodyAt(0).code).toBe(
      ': &\nwait "$!"\nset -euo pipefail\nlookup_user "{}"'
    );
  });

  it('describes the PTC timeout as a single sandbox run cap', () => {
    const schema = createCodeApiRunTimeoutSchema(15000);

    expect(clampCodeApiRunTimeoutMs(60000, 15000)).toBe(15000);
    expect(schema.default).toBe(15000);
    expect(schema.maximum).toBe(MAX_CODE_API_RUN_TIMEOUT_SCHEMA_MS);
    expect(schema.description).toContain('one sandbox run');
    expect(schema.description).toContain('not the total multi-round-trip');
    expect(schema.description).toContain('clamped before execution');
    expect(schema.description).toContain('Configured cap: 15 seconds');
  });

  it('keeps local programmatic timeout schemas aligned with local execution defaults', () => {
    const pythonTimeout = timeoutSchemaForTest(
      createLocalProgrammaticToolCallingTool().schema
    );
    const bashTimeout = timeoutSchemaForTest(
      createLocalBashProgrammaticToolCallingTool().schema
    );
    const configuredTimeout = timeoutSchemaForTest(
      createLocalProgrammaticToolCallingTool({ timeoutMs: 120000 }).schema
    );

    expect(pythonTimeout.default).toBe(60000);
    expect(pythonTimeout.maximum).toBe(300000);
    expect(pythonTimeout.description).toContain('local execution time');
    expect(bashTimeout.default).toBe(60000);
    expect(bashTimeout.maximum).toBe(300000);
    expect(configuredTimeout.default).toBe(120000);
    expect(configuredTimeout.maximum).toBe(300000);
  });

  it('forwards Authorization for bash programmatic requests', async () => {
    const tool = createBashProgrammaticToolCallingTool({
      authHeaders: { Authorization: 'Bearer bash-ptc-token' },
      baseUrl: 'https://code-stateful.example.com',
      executionProfile: 'stateful',
      runtimeSessionHint: 'user-123',
    });

    await tool.invoke(
      { code: 'lookup_user "{}"' },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer bash-ptc-token',
          'X-CodeAPI-Expected-Profile': 'stateful',
        }),
      })
    );
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://code-stateful.example.com/exec/programmatic'
    );
    expect(requestBodyAt(0).runtime_session_hint).toBe('user-123');
  });

  it('normalizes JSON-looking bash programmatic tool results before continuation', async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          status: 'tool_call_required',
          continuation_token: 'continue_123',
          tool_calls: [{ id: 'call_001', name: 'lookup_user', input: {} }],
        })
      )
      .mockResolvedValueOnce(completedResponse('done'));
    const tool = createBashProgrammaticToolCallingTool();
    const customToolMap = new Map([
      [
        'lookup_user',
        {
          name: 'lookup_user',
          invoke: jest.fn(async () =>
            JSON.stringify({
              result: {
                data: [{ id: 'user_123', name: 'Ada' }],
              },
            })
          ),
        },
      ],
    ]) as unknown as t.ToolMap;

    await tool.invoke(
      { code: 'lookup_user "{}"' },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: customToolMap,
          toolDefs,
        },
      }
    );

    expect(requestBodyAt(1).tool_results).toEqual([
      {
        call_id: 'call_001',
        result: {
          result: {
            data: [{ id: 'user_123', name: 'Ada' }],
          },
        },
        is_error: false,
      },
    ]);
  });

  it('reminds that failed bash programmatic executions do not register new files', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'error',
        error: 'jq failed',
        stderr: 'jq: Cannot index string with string "name"',
      })
    );
    const tool = createBashProgrammaticToolCallingTool();

    await expect(
      tool.invoke(
        {
          code: [
            'lookup_user "{}" > /mnt/data/user.json',
            'jq -r \'.result.name\' /mnt/data/user.json',
          ].join('\n'),
        },
        {
          toolCall: {
            name: 'bash_programmatic_code_execution',
            args: {},
            toolMap: toolMap(),
            toolDefs,
          },
        }
      )
    ).rejects.toThrow(
      'files written during this failed call were not registered for later calls'
    );
  });

  it('fetches session files with the CodeAPI resource scope and auth headers', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse([
        {
          id: 'file-1',
          resource_id: 'skill-1',
          storage_session_id: 'session_123',
          name: 'skill/file.txt',
          kind: 'skill',
          version: 7,
        },
      ])
    );

    const files = await fetchSessionFiles(
      'https://code.example.com',
      'session_123',
      { kind: 'skill', id: 'skill-1', version: 7 },
      undefined,
      { Authorization: 'Bearer files-token' }
    );

    expect(files).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/files/session_123?detail=full&kind=skill&id=skill-1&version=7',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer files-token',
        }),
      })
    );
  });

  it('fetches scoped session files with auth headers and no proxy placeholder', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await fetchSessionFiles(
      'https://code.example.com',
      'session_123',
      { kind: 'skill', id: 'skill-1', version: 7 },
      { Authorization: 'Bearer scoped-files-token' }
    );

    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/files/session_123?detail=full&kind=skill&id=skill-1&version=7',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer scoped-files-token',
        }),
      })
    );
  });

  it('preserves the legacy fetchSessionFiles proxy/auth argument order', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse([
        {
          name: 'session_123/file-1.txt',
          metadata: { 'original-filename': 'file.txt' },
        },
      ])
    );

    const files = await fetchSessionFiles(
      'https://code.example.com',
      'session_123',
      '',
      { Authorization: 'Bearer legacy-files-token' }
    );

    expect(files).toEqual([
      {
        storage_session_id: 'session_123',
        kind: 'user',
        id: 'file-1',
        resource_id: 'file-1',
        name: 'file.txt',
      },
    ]);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/files/session_123?detail=full',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer legacy-files-token',
        }),
      })
    );
  });
});

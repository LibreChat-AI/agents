import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import type * as t from '@/types';

/** Wire body the programmatic tools post, for both sandbox routes. */
type SandboxRequestBody = {
  lang?: 'bash' | 'py';
  code?: string;
  tools?: t.LCTool[];
  timeout?: number;
  session_id?: string;
  files?: t.CodeEnvFile[];
  runtime_session_hint?: string;
  continuation_token?: string;
};

type CapturedRequest = { url: string; body: SandboxRequestBody };

/* Mirrors codeapi's `/exec/programmatic` guard, which rejects an empty tool
 * manifest outright: `if (!tools || tools.length === 0) -> 400`. Plain `/exec`
 * has no such requirement. */
const requests: CapturedRequest[] = [];

jest.mock('node-fetch', () => ({
  __esModule: true,
  default: async (url: string, init?: { body?: string }) => {
    const body = JSON.parse(init?.body ?? '{}') as SandboxRequestBody;
    requests.push({ url, body });
    const truncation = body.code?.includes('trigger_truncation') === true
      ? {
        artifact_truncation: {
          code: 'artifact_truncated',
          reasons: { max_files: 1 },
          skipped: ['missing.csv'],
          skipped_count: 1,
        },
      }
      : {};

    if (url.endsWith('/exec/programmatic')) {
      if (!Array.isArray(body.tools) || body.tools.length === 0) {
        return {
          ok: false,
          status: 400,
          text: async () =>
            '{"error":"Missing required field: tools (must be a non-empty array)"}',
        };
      }
      return {
        ok: true,
        json: async () => ({
          status: 'completed',
          session_id: 'programmatic-session',
          stdout: 'programmatic',
          stderr: '',
          files: [],
          ...truncation,
        }),
      };
    }

    return {
      ok: true,
      json: async () => ({
        session_id: 'plain-session',
        stdout: 'plain',
        stderr: '',
        files: [],
        ...truncation,
      }),
    };
  },
}));

import {
  createProgrammaticToolCallingTool,
  wrapPythonForPlainExecution,
  normalizeToPythonIdentifier,
} from '../ProgrammaticToolCalling';
import { createBashProgrammaticToolCallingTool } from '../BashProgrammaticToolCalling';
import { assertUnambiguousIdentifiers } from '../ProgrammaticCallerPolicy';
import {
  createProgrammaticToolRegistry,
  createGetWeatherTool,
  createCalculatorTool,
} from '@/test/mockTools';

const BASE_URL = 'https://code.example.test';

const allToolDefs = Array.from(createProgrammaticToolRegistry().values());
const toolMap: t.ToolMap = new Map([
  ['get_weather', createGetWeatherTool()],
  ['calculator', createCalculatorTool()],
]);

/** Model-authored arguments accepted by both programmatic tools. */
type ProgrammaticArgs = {
  code: string;
  tool_manifest?: string[];
  timeout?: number;
};

/** Context ToolNode injects onto the tool call. */
type InjectedContext = {
  toolMap?: t.ToolMap;
  toolDefs?: t.LCTool[];
  disallowedToolDefs?: t.LCTool[];
  session_id?: string;
  _injected_files?: t.CodeEnvFile[];
  _runtime_session_hint?: string;
};

type ToolResult = {
  content: string;
  artifact?: t.ProgrammaticExecutionArtifact;
};

const DEFAULT_CONTEXT: InjectedContext = { toolMap, toolDefs: allToolDefs };

function invoke(
  create: typeof createBashProgrammaticToolCallingTool,
  name: string,
  args: ProgrammaticArgs,
  context: InjectedContext = DEFAULT_CONTEXT
): Promise<unknown> {
  return create({ baseUrl: BASE_URL }).invoke(
    { name, args, id: 'call-1', type: 'tool_call', ...context } as never,
    {} as never
  );
}

const invokeBash = (
  args: ProgrammaticArgs,
  context?: InjectedContext
): Promise<unknown> =>
  invoke(
    createBashProgrammaticToolCallingTool,
    'run_tools_with_bash',
    args,
    context
  );

const invokePython = (
  args: ProgrammaticArgs,
  context?: InjectedContext
): Promise<unknown> =>
  invoke(
    createProgrammaticToolCallingTool,
    'run_tools_with_code',
    args,
    context
  );

describe('an empty tool manifest runs instead of being rejected', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('routes bash to plain /exec', async () => {
    const result = (await invokeBash({
      code: 'ls /mnt/data',
      tool_manifest: [],
    })) as ToolResult;

    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
    expect(requests[0].body).toMatchObject({
      lang: 'bash',
      code: 'ls /mnt/data',
    });
    expect(requests[0].body.tools).toBeUndefined();
    expect(result.content).toContain('plain');
  });

  it('sends raw bash, without the replay `$!` guard', async () => {
    await invokeBash({ code: 'set -u\necho hi', tool_manifest: [] });

    expect(requests[0].body.code).toBe('set -u\necho hi');
  });

  it('routes python to plain /exec, keeping the async wrapper', async () => {
    await invokePython({
      code: 'import asyncio\nawait asyncio.sleep(0)\nprint("done")',
      tool_manifest: [],
    });

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
    expect(requests[0].body.lang).toBe('py');
    const sent = requests[0].body.code as string;
    expect(sent).toContain('async def __user_main__():');
    expect(sent).toContain('    await asyncio.sleep(0)');
    expect(sent).toContain('asyncio.run(__user_main__())');
  });

  it('preserves blank lines when wrapping python', () => {
    expect(wrapPythonForPlainExecution('a = 1\n\nprint(a)')).toContain(
      '    a = 1\n\n    print(a)'
    );
  });

  it('carries session, files, hint and timeout onto the plain path', async () => {
    const files: t.CodeEnvFile[] = [
      {
        kind: 'user',
        id: 'file-1',
        resource_id: 'user-1',
        name: 'data.csv',
        storage_session_id: 'prior',
      },
    ];

    const result = (await invokeBash(
      {
        code: 'wc -l /mnt/data/data.csv',
        tool_manifest: [],
        timeout: 5000,
      },
      {
        ...DEFAULT_CONTEXT,
        session_id: 'prior',
        _injected_files: files,
        _runtime_session_hint: 'hint-1',
      }
    )) as ToolResult;

    expect(requests[0].body).toMatchObject({
      session_id: 'prior',
      files,
      runtime_session_hint: 'hint-1',
      timeout: 5000,
    });
    expect(result.artifact?.session_id).toBe('plain-session');
  });
});

describe('artifact truncation across programmatic routes', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it.each([
    ['bash', invokeBash, []],
    ['python', invokePython, []],
    ['bash with tools', invokeBash, ['get_weather']],
    ['python with tools', invokePython, ['get_weather']],
  ] as const)('surfaces omissions for %s', async (_kind, invokeTool, tool_manifest) => {
    const result = (await invokeTool({
      code: _kind.startsWith('python')
        ? 'print("trigger_truncation")'
        : 'echo trigger_truncation',
      tool_manifest: [...tool_manifest],
    })) as ToolResult;

    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe(
      `${BASE_URL}/${tool_manifest.length > 0 ? 'exec/programmatic' : 'exec'}`
    );
    expect(result.content).toContain('1 file(s) were omitted from delivery');
    expect(result.content).toContain('missing.csv');
    expect(result.content).toContain('do not rerun automatically');
    expect(result.artifact?.artifact_truncation).toEqual({
      code: 'artifact_truncated',
      reasons: { max_files: 1 },
      skipped: ['missing.csv'],
      skipped_count: 1,
    });
  });
});

describe('behavior that stays as it was', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('still requires a manifest when direct-only tools exist', async () => {
    await expect(
      invokeBash(
        { code: 'ls /mnt/data' },
        { ...DEFAULT_CONTEXT, disallowedToolDefs: [{ name: 'send_email' }] }
      )
    ).rejects.toThrow('requires a tool_manifest');

    expect(requests).toHaveLength(0);
  });

  it('still refuses a manifest naming a direct-only tool', async () => {
    await expect(
      invokeBash(
        { code: 'send_email \'{}\'', tool_manifest: ['send_email'] },
        { ...DEFAULT_CONTEXT, disallowedToolDefs: [{ name: 'send_email' }] }
      )
    ).rejects.toThrow('not marked for code_execution');
  });

  it('still sends the full manifest when none was requested', async () => {
    await invokeBash({ code: 'ls /mnt/data' });

    expect(requests[0].url).toBe(`${BASE_URL}/exec/programmatic`);
    expect(requests[0].body.tools?.map((def) => def.name)).toEqual(
      allToolDefs.map((def) => def.name)
    );
  });

  it('still reports a missing toolMap when nothing was injected', async () => {
    await expect(
      createBashProgrammaticToolCallingTool({ baseUrl: BASE_URL }).invoke({
        code: 'get_weather \'{}\'',
      } as never)
    ).rejects.toThrow('No toolMap provided');

    expect(requests).toHaveLength(0);
  });

  it('runs tool-free code when ToolNode injects an empty context', async () => {
    await invokeBash(
      { code: 'ls /mnt/data' },
      { toolMap: new Map(), toolDefs: [] }
    );

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
  });
});

describe('ambiguous runtime identifiers', () => {
  const colliding: t.LCTool[] = [
    { name: 'report-tool', allowed_callers: ['code_execution'] },
    { name: 'report_tool', allowed_callers: ['code_execution'] },
  ];

  beforeEach(() => {
    requests.length = 0;
  });

  it('rejects a stub set holding both', () => {
    expect(() =>
      assertUnambiguousIdentifiers(
        colliding,
        normalizeToPythonIdentifier,
        'run_tools_with_code'
      )
    ).toThrow('cannot tell which one the code means');
  });

  it('accepts a stub set holding one of the pair', () => {
    expect(() =>
      assertUnambiguousIdentifiers(
        [colliding[0]],
        normalizeToPythonIdentifier,
        'run_tools_with_code'
      )
    ).not.toThrow();
  });

  it('rejects before the request when the fallback is ambiguous', async () => {
    const toolCallMap: t.ToolMap = new Map([
      ['report-tool', createCalculatorTool()],
    ]);

    await expect(
      invokeBash(
        { code: 'report_tool \'{}\'' },
        { toolMap: toolCallMap, toolDefs: colliding }
      )
    ).rejects.toThrow('cannot tell which one the code means');

    expect(requests).toHaveLength(0);
  });

  it('does not reject a tool-free call that never emits stubs', async () => {
    await invokeBash(
      { code: 'ls /mnt/data', tool_manifest: [] },
      { toolMap: new Map(), toolDefs: colliding }
    );

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
  });
});

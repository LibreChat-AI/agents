import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, it, expect, jest } from '@jest/globals';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  createLocalProgrammaticToolCallingTool,
  createLocalBashProgrammaticToolCallingTool,
} from '../local/LocalProgrammaticToolCalling';
import { normalizeToBashIdentifier } from '../BashProgrammaticToolCalling';
import { normalizeToPythonIdentifier } from '../ProgrammaticToolCalling';
import { selectProgrammaticTools } from '../ProgrammaticCallerPolicy';

const registeredName = 'stub_echo_mcp_hypen-server';
const sandboxName = 'stub_echo_mcp_hypen_server';
const toolDef: t.LCTool = {
  name: registeredName,
  allowed_callers: ['code_execution'],
};

it.each([
  ['Bash runner', createLocalBashProgrammaticToolCallingTool, 'bash'],
  [
    'code runner, Bash default',
    createLocalProgrammaticToolCallingTool,
    undefined,
  ],
  ['code runner, Python', createLocalProgrammaticToolCallingTool, 'py'],
])(
  '%s calls the registered tool through its sandbox name',
  async (_label, createRunner, lang) => {
    const echo = tool(async ({ message }) => message, {
      name: registeredName,
      description: 'Echo a message',
      schema: z.object({ message: z.string() }),
    });
    const invokeSpy = jest.spyOn(echo, 'invoke');
    const runner = createRunner();
    const toolCall = {
      id: 'echo-call',
      name: runner.name,
      args: {},
      type: 'tool_call',
      toolDefs: [toolDef],
      toolMap: new Map([[registeredName, echo]]),
      disallowedToolDefs: [{ name: 'direct_only' }],
    } satisfies ToolCall & Partial<t.ProgrammaticCache>;
    const result = await runner.invoke(
      {
        ...(lang == null ? {} : { lang }),
        tool_manifest: [sandboxName],
        code:
          lang === 'py'
            ? `print(await ${sandboxName}(message="hello from hyphenated server"))`
            : `${sandboxName} '{"message":"hello from hyphenated server"}'`,
      },
      { toolCall }
    );

    expect(result).toMatchObject({
      content: expect.stringContaining('hello from hyphenated server'),
    });
    expect(invokeSpy).toHaveBeenCalledTimes(1);
    expect(echo.name).toBe(registeredName);
    expect(invokeSpy.mock.calls[0][0]).toEqual({
      message: 'hello from hyphenated server',
    });
  }
);

it.each([
  ['Bash', normalizeToBashIdentifier],
  ['Python', normalizeToPythonIdentifier],
])(
  '%s accepts the sandbox name in a mixed-caller manifest',
  (_runtime, normalizeIdentifier) => {
    expect(
      selectProgrammaticTools({
        requestedToolNames: [sandboxName],
        allowedToolDefs: [toolDef],
        disallowedToolDefs: [{ name: 'direct_only' }],
        programmaticToolName: 'runner',
        normalizeIdentifier,
      })
    ).toEqual([toolDef]);
  }
);

describe.each([
  ['Bash', normalizeToBashIdentifier, 'if'],
  ['Python', normalizeToPythonIdentifier, 'return'],
])('%s sandbox-name resolution', (_runtime, normalizeIdentifier, keyword) => {
  const select = (
    requestedToolNames: string[],
    allowedToolDefs = [toolDef],
    disallowedToolDefs: t.LCTool[] = []
  ) =>
    selectProgrammaticTools({
      requestedToolNames,
      allowedToolDefs,
      disallowedToolDefs,
      programmaticToolName: 'runner',
      normalizeIdentifier,
    });

  it.each(['my-tool', 'my tool', 'my.tool', '1tool', keyword])(
    'resolves %s using the runtime normalizer',
    (name) => {
      const definition = { name };
      expect(select([normalizeIdentifier(name)], [definition])).toEqual([
        definition,
      ]);
    }
  );

  it('deduplicates registered and sandbox names in request order', () => {
    const other = { name: 'other' };
    expect(
      select([sandboxName, 'other', registeredName], [toolDef, other])
    ).toEqual([toolDef, other]);
  });

  it('keeps registered names backwards compatible', () => {
    expect(select([registeredName])).toEqual([toolDef]);
  });

  it('rejects normalized direct-only names', () => {
    expect(() =>
      select(['send_email'], [toolDef], [{ name: 'send-email' }])
    ).toThrow('not marked for code_execution');
  });

  it('rejects an alias shared with a direct-only tool', () => {
    expect(() =>
      select(['a_b_c'], [{ name: 'a-b_c' }], [{ name: 'a_b-c' }])
    ).toThrow('not marked for code_execution');
  });

  it('rejects ambiguous aliases instead of choosing by declaration order', () => {
    expect(() =>
      select(['a_b_c'], [{ name: 'a-b_c' }, { name: 'a_b-c' }])
    ).toThrow('cannot tell which one the code means');
  });

  it('prefers an exact registered name over another tool’s alias', () => {
    const exact = { name: sandboxName };
    expect(select([sandboxName], [toolDef, exact])).toEqual([exact]);
  });

  it('allows explicit selection of one tool from a colliding pair', () => {
    expect(select([registeredName], [toolDef, { name: sandboxName }])).toEqual([
      toolDef,
    ]);
  });

  it('rejects unknown names', () => {
    expect(() => select(['missing_tool'])).toThrow(
      'not available for code_execution'
    );
  });

  it('does not normalize arbitrary manifest entries into registered names', () => {
    expect(() => select(['stub-echo_mcp_hypen-server'])).toThrow(
      'not available for code_execution'
    );
  });
});

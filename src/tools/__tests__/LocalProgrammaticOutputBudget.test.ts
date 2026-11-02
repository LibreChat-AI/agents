import { tmpdir } from 'os';
import { join } from 'path';
import { spawn } from 'child_process';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { LocalExecutionConfig, LocalSpawn } from '@/types';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from '../local/LocalProgrammaticToolCalling';
import {
  _convertMessagesToOpenAIParams,
  _convertMessagesToOpenAIResponsesParams,
} from '@/llm/openai/utils';
import { createLocalProgrammaticProfile } from '../local/LocalProgrammaticProfile';
import { Constants } from '@/common';
import { ToolNode } from '../ToolNode';

const runtimes = ['python', 'bash', 'bash-via-code'] as const;
type Runtime = (typeof runtimes)[number];

describe('local programmatic observation budget', () => {
  let cwd: string;
  let callIndex = 0;
  const captures = new Set<string>();

  beforeEach(async () => {
    cwd = await mkdtemp(join(tmpdir(), 'lc-ptc-output-budget-'));
  });
  afterEach(async () => {
    await Promise.all([...captures].map((path) => rm(path, { force: true })));
    captures.clear();
    await rm(cwd, { recursive: true, force: true });
  });

  function capturePath(output: string): string {
    const path = /^full_output_path: (.+) \(/m.exec(output)?.[1];
    if (path == null) throw new Error('Missing output capture');
    captures.add(path);
    return path;
  }

  function runner(runtime: Runtime, local: LocalExecutionConfig) {
    return runtime === 'bash'
      ? createLocalBashProgrammaticToolCallingTool(local)
      : createLocalProgrammaticToolCallingTool(local);
  }

  async function execute(node: ToolNode, name: string, args: { code: string; lang: string; tool_manifest: string[] } | { command: string; args: string[] }) {
    const callId = `call-budget-${callIndex++}`;
    const call = new AIMessage({ content: '', tool_calls: [{ id: callId, name, args }] });
    const result = await node.invoke({ messages: [call] }) as { messages: ToolMessage[] };
    expect(result.messages).toHaveLength(1);
    const message = result.messages[0];
    expect(message.tool_call_id).toBe(callId);
    if (typeof message.content !== 'string') throw new Error('Expected text output');
    if (message.content.includes('full_output_path:')) capturePath(message.content);
    return { call, message, output: message.content };
  }

  it.each(runtimes)('keeps nested data intact and recovers omitted %s output through the same tool surface', async (runtime) => {
    const payload = JSON.stringify(Array.from({ length: 80 }, (_, id) => ({ id, value: 'data'.repeat(12) })));
    await writeFile(join(cwd, 'records.json'), payload);
    const local: LocalExecutionConfig = {
      cwd, toolResultMaxChars: 12_000, programmaticResultMaxChars: 256,
      maxOutputChars: 20_000, sandbox: { enabled: false },
      codingToolNames: ['read_file', 'bash_tool'],
    };
    const profile = createLocalProgrammaticProfile({ mode: 'hybrid', local });
    const node = new ToolNode({
      ...profile,
      toolExecution: {
        ...profile.toolExecution,
        local: { ...profile.toolExecution.local, codingToolNames: ['read_file', 'bash_tool', Constants.PROGRAMMATIC_TOOL_CALLING, Constants.BASH_PROGRAMMATIC_TOOL_CALLING] },
      },
    });
    const name = runtime === 'bash' ? Constants.BASH_PROGRAMMATIC_TOOL_CALLING : Constants.PROGRAMMATIC_TOOL_CALLING;
    const code = runtime === 'python'
      ? 'import json\nraw = await read_file(path="records.json", line_numbers=False)\nrows = json.loads(raw)\nprint(raw)\nprint("selected=" + str(rows[-1]["id"]))'
      : 'raw=$(read_file \'{"path":"records.json","line_numbers":false}\'); printf "%s\\n" "$raw"; printf "%s" "$raw" | python3 -c \'import json,sys; print("selected=" + str(json.load(sys.stdin)[-1]["id"]))\'';
    const { call, message, output } = await execute(node, name, {
      lang: runtime === 'python' ? 'py' : 'bash', code, tool_manifest: ['read_file'],
    });
    expect(message.status).toBe('success');
    expect(output).toContain('selected=79');
    expect(output).toContain('output_truncated: true');
    expect(output.length).toBeLessThan(1200);
    const path = capturePath(output);
    expect(local.toolResultMaxChars).toBe(12_000);

    const recovered = await execute(node, Constants.BASH_TOOL, {
      command: 'python3 -c \'import json,sys; f=open(sys.argv[1]); rows=json.loads(f.readline()); print("recovered=" + str(rows[40]["id"]))\' "$1"',
      args: [path],
    });
    expect(recovered.message.status).toBe('success');
    expect(recovered.output).toContain('recovered=40');

    const history = [new HumanMessage('Process the local records.'), call, message];
    for (const convert of [_convertMessagesToOpenAIParams, _convertMessagesToOpenAIResponsesParams]) {
      const first = convert(history);
      const firstJson = JSON.stringify(first);
      expect(firstJson).not.toContain(payload);
      expect(firstJson).toContain('selected=79');
      const continued = convert([...history, recovered.call, recovered.message]);
      expect(JSON.stringify(continued.slice(0, first.length))).toBe(firstJson);
      expect(message.content).toBe(output);
    }

    await rm(path);
    const missing = await execute(node, Constants.BASH_TOOL, { command: 'cat "$1"', args: [path] });
    expect(missing.output).toContain('exit_code: 1');
    expect(missing.output).not.toContain('recovered=40');
  });

  it.each(runtimes)('preserves inherited limits, explicit zero and failure evidence for %s', async (runtime) => {
    for (const programmaticResultMaxChars of [undefined, 0, 256]) {
      const local: LocalExecutionConfig = {
        cwd, toolResultMaxChars: 2000, programmaticResultMaxChars,
        maxOutputChars: 20_000, sandbox: { enabled: false },
      };
      const tool = runner(runtime, local);
      const invoke = (fail: boolean) => tool.invoke({
        lang: runtime === 'python' ? 'py' : 'bash', tool_manifest: [],
        code: runtime === 'python'
          ? `import sys\nprint('x' * 3000)\nprint('failure-evidence', file=sys.stderr)\nsys.exit(${fail ? 7 : 0})`
          : `printf '%03000d' 0; printf failure-evidence >&2; exit ${fail ? 7 : 0}`,
      }, { toolCall: { name: tool.name, args: {}, toolMap: new Map(), toolDefs: [] } });
      const output = String(await invoke(false));
      if (output.includes('full_output_path:')) capturePath(output);
      expect(output).toContain('failure-evidence');
      if (programmaticResultMaxChars === 0) {
        expect(output).not.toContain('output_truncated: true');
        expect(output).not.toContain('full_output_path:');
      } else {
        capturePath(output);
        expect(output).toContain('output_truncated: true');
        expect(output.length).toBeLessThan((programmaticResultMaxChars ?? 2000) + 900);
      }
      try {
        await invoke(true);
        throw new Error('Expected failed program');
      } catch (error) {
        if (!(error instanceof Error)) throw error;
        if (error.message.includes('full_output_path:')) capturePath(error.message);
        expect(error.message).toContain('exit_code: 7');
        expect(error.message).toContain('failure-evidence');
        if (programmaticResultMaxChars !== 0) capturePath(error.message);
      }
    }
  });

  it.each([NaN, Infinity, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])('rejects an invalid outer budget before execution: %s', (programmaticResultMaxChars) => {
    for (const create of [createLocalProgrammaticToolCallingTool, createLocalBashProgrammaticToolCallingTool]) {
      expect(() => create({ programmaticResultMaxChars })).toThrow('programmaticResultMaxChars');
    }
  });

  it.each([0, 4000])('retains the independent engine limit with outer budget %i', async (programmaticResultMaxChars) => {
    const tool = runner('python', {
      cwd, programmaticResultMaxChars, toolResultMaxChars: 12_000,
      maxOutputChars: 256, sandbox: { enabled: false },
    });
    const output = String(await tool.invoke({
      lang: 'py', code: 'print("x" * 3000)', tool_manifest: [],
    }, { toolCall: { name: tool.name, args: {}, toolMap: new Map(), toolDefs: [] } }));
    capturePath(output);
    expect(output).toContain('output_truncated: true');
    expect(output.length).toBeLessThan(1200);
  });

  it.each(['exec', 'spawn'] as const)('discloses host-only capture for a custom %s launcher', async (kind) => {
    const launch: LocalSpawn = (command, args, options) => spawn(command, args, { ...options, stdio: 'pipe' });
    const tool = runner('python', {
      cwd, programmaticResultMaxChars: 256, sandbox: { enabled: false },
      ...(kind === 'exec' ? { exec: { spawn: launch } } : { spawn: launch }),
    });
    const output = String(await tool.invoke({
      lang: 'py', code: 'print("x" * 3000)', tool_manifest: [],
    }, { toolCall: { name: tool.name, args: {}, toolMap: new Map(), toolDefs: [] } }));
    capturePath(output);
    expect(output).toContain('host capture; retrieval requires the host filesystem');
    expect(output).not.toContain('use local bash to inspect');
  });

  it('allows an outer budget larger than the inherited combined limit', async () => {
    const local: LocalExecutionConfig = {
      cwd, programmaticResultMaxChars: 2000, toolResultMaxChars: 256,
      sandbox: { enabled: false },
    };
    const tool = runner('python', local);
    const output = String(await tool.invoke({
      lang: 'py', code: 'print("x" * 1000)', tool_manifest: [],
    }, { toolCall: { name: tool.name, args: {}, toolMap: new Map(), toolDefs: [] } }));
    if (output.includes('full_output_path:')) capturePath(output);
    expect(output).toContain('x'.repeat(1000));
    expect(output).not.toContain('output_truncated: true');
    expect(local.toolResultMaxChars).toBe(256);
  });
});

import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, readFile, rm } from 'fs/promises';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from '../local/LocalProgrammaticToolCalling';
import { ProgrammaticToolCallingDefinition } from '../ProgrammaticToolCalling';
import { BashProgrammaticToolCallingDefinition } from '../BashProgrammaticToolCalling';

describe('local programmatic workspace contract', () => {
  it.each([
    createLocalProgrammaticToolCallingTool,
    createLocalBashProgrammaticToolCallingTool,
  ])('describes the actual local filesystem in both model-facing surfaces', (createTool) => {
    const runner = createTool();
    const schema = runner.schema as { properties: { code: { description: string } } };
    for (const guidance of [runner.description, schema.properties.code.description]) {
      expect(guidance).toContain('configured working directory');
      expect(guidance).toContain('Project files persist');
      expect(guidance).not.toContain('/mnt/data');
      expect(guidance).not.toContain('replay');
      expect(guidance).not.toContain('Failed executions register nothing');
    }
  });

  it('keeps the remote artifact contract on the remote definitions', () => {
    for (const definition of [ProgrammaticToolCallingDefinition, BashProgrammaticToolCallingDefinition]) {
      expect(definition.description).toContain('/mnt/data');
      expect(definition.schema.properties.code.description).toContain('/mnt/data');
    }
  });

  it.each(['python', 'bash', 'bash-via-code'] as const)(
    'preserves relative workspace files across successful and failed %s calls',
    async (runtime) => {
      const cwd = await mkdtemp(join(tmpdir(), 'lc-ptc-workspace-'));
      const options = { cwd, sandbox: { enabled: false } };
      const runner = runtime === 'bash'
        ? createLocalBashProgrammaticToolCallingTool(options)
        : createLocalProgrammaticToolCallingTool(options);
      const invoke = (code: string) => runner.invoke({
        lang: runtime === 'python' ? 'py' : 'bash', code, tool_manifest: [],
      }, { toolCall: { name: runner.name, args: {}, toolMap: new Map(), toolDefs: [] } });
      try {
        await invoke(runtime === 'python'
          ? 'from pathlib import Path\nPath("snapshot.json").write_text("saved")'
          : 'printf saved > snapshot.json');
        expect(await readFile(join(cwd, 'snapshot.json'), 'utf8')).toBe('saved');
        const next = await invoke(runtime === 'python'
          ? 'from pathlib import Path\nprint(Path("snapshot.json").read_text())'
          : 'cat snapshot.json');
        expect(next).toContain('saved');
        await expect(invoke(runtime === 'python'
          ? 'from pathlib import Path\nPath("checkpoint.json").write_text("partial")\nraise RuntimeError("later failure")'
          : 'printf partial > checkpoint.json; exit 1')).rejects.toThrow();
        expect(await readFile(join(cwd, 'checkpoint.json'), 'utf8')).toBe('partial');
      } finally {
        await rm(cwd, { recursive: true, force: true });
      }
    }
  );
});

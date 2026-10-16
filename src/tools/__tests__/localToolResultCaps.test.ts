import { tmpdir } from 'os';
import { join } from 'path';
import { existsSync, readFileSync, statSync, unlinkSync } from 'fs';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { describe, it, expect, afterEach } from '@jest/globals';
import {
  createLocalGrepSearchTool,
  createLocalReadFileTool,
} from '../local/LocalCodingTools';
import {
  capHeadTail,
  capLines,
  spillToFile,
  tailTruncate,
} from '../local/toolResultCaps';
import { createLocalBashExecutionTool } from '../local/LocalExecutionTools';

const tempDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'lc-caps-'));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true }))
  );
});

describe('spillToFile', () => {
  it('writes the full text to a file only its owner can read', () => {
    const path = spillToFile('full output\n');
    try {
      expect(readFileSync(path, 'utf8')).toBe('full output\n');
      if (process.platform !== 'win32') {
        expect(statSync(path).mode & 0o777).toBe(0o600);
      }
    } finally {
      if (existsSync(path)) {
        unlinkSync(path);
      }
    }
  });
});

describe('tailTruncate', () => {
  it('starts the kept tail at a line boundary when one is near', () => {
    expect(tailTruncate('first line\nsecond line\n', 15)).toBe('second line\n');
  });
});

describe('toolResultMaxChars', () => {
  it('caps search lines and counts what it left out', () => {
    const long = `a.py:1:${'x'.repeat(500)}`;
    const capped = capLines([long, 'b.py:2:y', 'c.py:3:z'].join('\n'), 320);
    expect(capped.split('\n')[0]).toMatch(/\[line truncated\]$/);
    expect(capped).toMatch(
      /\[2 more matching lines not shown; narrow the pattern or path\]$/
    );
  });

  it('keeps the head and tail of long shell output and names the full file', () => {
    const text = `HEAD\n${'m\n'.repeat(2000)}TAIL summary\n`;
    const capped = capHeadTail(text, 400, () => '/tmp/full.txt');
    expect(capped.startsWith('HEAD')).toBe(true);
    expect(capped).toContain('full output at /tmp/full.txt');
    expect(capped.trimEnd().endsWith('TAIL summary')).toBe(true);
    expect(capped.length).toBeLessThan(600);
    expect(capHeadTail('short', 400)).toBe('short');
  });

  it('caps read_file at whole lines with an offset hint, and leaves defaults alone', async () => {
    const cwd = await createTempDir();
    await writeFile(
      join(cwd, 'big.py'),
      Array.from({ length: 500 }, (_, i) => `value_${i} = ${i}`).join('\n')
    );
    const capped = String(
      await createLocalReadFileTool({ cwd, toolResultMaxChars: 1000 }).invoke({
        path: 'big.py',
      })
    );
    expect(capped.length).toBeLessThan(1150);
    expect(capped).toMatch(
      /\[showing lines 1-\d+; continue with offset=\d+\]$/
    );
    const next = Number(/offset=(\d+)/.exec(capped)?.[1]);
    const rest = String(
      await createLocalReadFileTool({ cwd, toolResultMaxChars: 1000 }).invoke({
        path: 'big.py',
        offset: next,
      })
    );
    expect(
      rest.startsWith(
        `${String(next).padStart(6, ' ')}\tvalue_${next - 1} = ${next - 1}`
      )
    ).toBe(true);
    expect(
      String(await createLocalReadFileTool({ cwd }).invoke({ path: 'big.py' }))
    ).toContain('value_499');
  });

  it('cuts a single line longer than the cap instead of returning it whole', async () => {
    const cwd = await createTempDir();
    await writeFile(join(cwd, 'min.json'), `[${'1,'.repeat(50000)}1]\n{}`);
    const capped = String(
      await createLocalReadFileTool({ cwd, toolResultMaxChars: 1000 }).invoke({
        path: 'min.json',
      })
    );
    expect(capped.length).toBeLessThan(1150);
    expect(capped).toContain('[line truncated]');
    expect(capped).toMatch(/continue with offset=2\]$/);
  });

  it('applies the cap to reads with an explicit limit too', async () => {
    const cwd = await createTempDir();
    await writeFile(
      join(cwd, 'wide.py'),
      Array.from({ length: 50 }, (_, i) => `v${i} = '${'x'.repeat(200)}'`).join(
        '\n'
      )
    );
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 1000 });
    const capped = String(await read.invoke({ path: 'wide.py', limit: 40 }));
    expect(capped.length).toBeLessThan(1150);
    expect(capped).toMatch(/continue with offset=\d+\]$/);
    const short = String(await read.invoke({ path: 'wide.py', limit: 2 }));
    expect(short).toMatch(/showing lines 1-2; continue with offset=3\]$/);
  });

  it('names no spill file when commands run through a delegated runtime', () => {
    const text = `${'a'.repeat(500)}\n${'b'.repeat(500)}`;
    const capped = capHeadTail(text, 400);
    expect(capped).toContain('full output not kept');
    expect(capped.length).toBeLessThan(500);
  });

  it('reads the last line of a file without a trailing newline and stops at the end', async () => {
    const cwd = await createTempDir();
    await writeFile(join(cwd, 'tail.txt'), 'one\ntwo\nthree');
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 1000 });
    expect(String(await read.invoke({ path: 'tail.txt', offset: 2 }))).toBe(
      '     2\ttwo\n     3\tthree'
    );
    expect(String(await read.invoke({ path: 'tail.txt', offset: 9 }))).toBe('');
  });

  it('keeps the exit status of a failing command under a tiny cap', async () => {
    const cwd = await createTempDir();
    const result = String(
      await createLocalBashExecutionTool({
        config: { cwd, toolResultMaxChars: 40 },
      }).invoke({ command: 'seq 1 5000; exit 3' })
    );
    expect(result).toMatch(/exit_code: 3/);
  });

  it('caps grep_search and bash_tool results only when set', async () => {
    const cwd = await createTempDir();
    await writeFile(
      join(cwd, 'm.py'),
      Array.from({ length: 400 }, (_, i) => `needle_${i} = 1`).join('\n')
    );
    const grep = String(
      await createLocalGrepSearchTool({ cwd, toolResultMaxChars: 500 }).invoke({
        pattern: 'needle',
      })
    );
    expect(grep).toMatch(/more matching lines not shown/);
    const plain = String(
      await createLocalGrepSearchTool({ cwd }).invoke({ pattern: 'needle' })
    );
    expect(plain).not.toMatch(/not shown/);
    const bash = String(
      await createLocalBashExecutionTool({
        config: { cwd, toolResultMaxChars: 500 },
      }).invoke({ command: 'seq 1 2000; echo DONE' })
    );
    expect(bash).toContain('full output at');
    expect(bash).toContain('DONE');
  });
});

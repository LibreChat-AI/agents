import { tmpdir } from 'os';
import { join } from 'path';
import type { SpawnOptions } from 'child_process';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { describe, it, expect, afterEach } from '@jest/globals';
import {
  createLocalGrepSearchTool,
  createLocalReadFileTool, _resetRipgrepCacheForTests
} from '../local/LocalCodingTools';
import {
  capHeadTail,
  capLines,
  tailTruncate,
} from '../local/toolResultCaps';
import { createLocalBashExecutionTool } from '../local/LocalExecutionTools';
import type { LocalSpawn } from '@/types';

const tempDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'lc-caps-'));
  tempDirs.push(dir);
  return dir;
}

function continuation(text: string): { body: string; cursor?: string } {
  const match = /\n\[continue with cursor=(v1:[a-f0-9]{64}:\d+)\]$/.exec(text);
  return match == null
    ? { body: text }
    : { body: text.slice(0, match.index), cursor: match[1] };
}

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true }))
  );
});

describe('tailTruncate', () => {
  it('starts the kept tail at a line boundary when one is near', () => {
    expect(tailTruncate('first line\nsecond line\n', 15)).toBe('second line\n');
    expect(capHeadTail(`${'x'.repeat(200)}FAILURE\n`, 40)).toContain('FAILURE');
  });
});

describe('toolResultMaxChars', () => {
  it('caps search lines and counts what it left out', () => {
    const long = `a.py:1:${'x'.repeat(500)}`;
    const capped = capLines([long, 'b.py:2:y', 'c.py:3:z'].join('\n'), 320);
    expect(capped.split('\n')[0]).toMatch(/\[line truncated\]$/);
    expect(capLines(long, 40).length).toBeLessThan(80);
    expect(capLines([long, 'b.py:2:y', 'c.py:3:z'], 320)).toBe(capped);
    expect(capped).toMatch(
      /\[2 more matching lines not shown; narrow the pattern or path\]$/
    );
  });

  it('keeps the head and tail of long shell output and counts what it left out', () => {
    const text = `HEAD\n${'m\n'.repeat(2000)}TAIL summary\n`;
    const capped = capHeadTail(text, 400);
    expect(capped.startsWith('HEAD')).toBe(true);
    expect(capped).toMatch(/\[\.\.\. \d+ characters omitted; rerun with narrower output/);
    expect(capped.trimEnd().endsWith('TAIL summary')).toBe(true);
    expect(capped.length).toBeLessThan(600);
    expect(capHeadTail('short', 400)).toBe('short');
  });

  it('preserves content-bound continuation for capped reads with explicit limits', async () => {
    const cwd = await createTempDir();
    await writeFile(
      join(cwd, 'wide.py'),
      Array.from({ length: 50 }, (_, i) => `v${i} = '${'x'.repeat(200)}'`).join(
        '\n'
      )
    );
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 1000 });
    const first = continuation(
      String(await read.invoke({ path: 'wide.py', limit: 40 }))
    );
    expect(first.body.length).toBeLessThanOrEqual(1000);
    expect(first.cursor).toBeDefined();
    const firstLineCount = (first.body.match(/^ *\d+\t/gm) ?? []).length;
    const afterFirst = String(
      await read.invoke({ path: 'wide.py', cursor: first.cursor })
    );
    expect(afterFirst).toMatch(
      new RegExp(`^ {5}${firstLineCount + 1}\\t`)
    );

    const limited = continuation(
      String(await read.invoke({ path: 'wide.py', limit: 2 }))
    );
    expect(limited.body).toMatch(/^ {5}1\t/);
    expect(limited.cursor).toBeDefined();
    expect(
      String(
        await read.invoke({ path: 'wide.py', cursor: limited.cursor })
      )
    ).toMatch(/^ {5}3\t/);
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

  it('keeps fallback grep skip notices when matches fill the cap', async () => {
    _resetRipgrepCacheForTests();
    const realSpawn = (
      require('child_process') as typeof import('child_process')
    ).spawn;
    const noRg = ((cmd: string, args: string[], opts: SpawnOptions) =>
      cmd === 'rg'
        ? realSpawn('sh', ['-c', 'exit 127'], opts)
        : realSpawn(cmd, args, opts)) as unknown as LocalSpawn;
    const cwd = await createTempDir();
    await writeFile(
      join(cwd, 'many.py'),
      Array.from({ length: 200 }, (_, i) => `needle_${i} = ${i}`).join('\n')
    );
    await writeFile(join(cwd, 'big.txt'), Buffer.alloc(6 * 1024 * 1024, 'a'));
    const capped = String(
      await createLocalGrepSearchTool({
        cwd,
        toolResultMaxChars: 300,
        exec: { spawn: noRg },
      }).invoke({ pattern: 'needle' })
    );
    expect(capped).toMatch(/more matching lines not shown/);
    expect(capped).toMatch(/skipped/);
  });

  it('caps fallback grep pattern errors', async () => {
    _resetRipgrepCacheForTests();
    const realSpawn = (
      require('child_process') as typeof import('child_process')
    ).spawn;
    const noRg = ((cmd: string, args: string[], opts: SpawnOptions) =>
      cmd === 'rg'
        ? realSpawn('sh', ['-c', 'exit 127'], opts)
        : realSpawn(cmd, args, opts)) as unknown as LocalSpawn;
    const cwd = await createTempDir();
    await writeFile(join(cwd, 'a.py'), 'x = 1\n');
    const refused = String(
      await createLocalGrepSearchTool({
        cwd,
        toolResultMaxChars: 120,
        exec: { spawn: noRg },
      }).invoke({ pattern: `(${'a'.repeat(900)}` })
    );
    expect(refused).toMatch(/^grep_search refused the pattern/);
    expect(refused.length).toBeLessThan(260);
  });

  it('bounds many fallback grep skip notices by the cap', async () => {
    _resetRipgrepCacheForTests();
    const realSpawn = (
      require('child_process') as typeof import('child_process')
    ).spawn;
    const noRg = ((cmd: string, args: string[], opts: SpawnOptions) =>
      cmd === 'rg'
        ? realSpawn('sh', ['-c', 'exit 127'], opts)
        : realSpawn(cmd, args, opts)) as unknown as LocalSpawn;
    const cwd = await createTempDir();
    const big = Buffer.alloc(6 * 1024 * 1024, 'a');
    for (let i = 0; i < 12; i++) {
      await writeFile(join(cwd, `big-file-with-a-long-name-${i}.txt`), big);
    }
    const capped = String(
      await createLocalGrepSearchTool({
        cwd,
        toolResultMaxChars: 400,
        exec: { spawn: noRg },
      }).invoke({ pattern: 'needle' })
    );
    expect(capped.length).toBeLessThan(500);
    expect(capped).toMatch(/more files skipped\]$/);
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
    expect(bash).toContain('characters omitted');
    expect(bash).toContain('DONE');
  });
});

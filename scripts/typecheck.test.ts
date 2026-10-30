import { test } from 'node:test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import type { TestContext } from 'node:test';

const runner = fileURLToPath(new URL('./typecheck.ts', import.meta.url));

async function fixture(t: TestContext, bun?: string): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'agents-typecheck-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'bin'));
  await writeFile(
    join(root, 'tsconfig.json'),
    JSON.stringify({
      compilerOptions: { strict: true, types: [], skipLibCheck: true },
    })
  );
  await writeFile(
    join(root, 'index.ts'),
    'export const value: string = "ok";\n'
  );
  if (bun !== undefined) {
    await writeFile(join(root, 'bin', 'bun'), `#!${process.execPath}\n${bun}`, {
      mode: 0o755,
    });
  }
  return root;
}

function run(
  root: string,
  args: string[] = [],
  script = runner
): ReturnType<typeof spawnSync> {
  return spawnSync(process.execPath, [script, ...args], {
    cwd: root,
    env: { ...process.env, PATH: join(root, 'bin') },
    encoding: 'utf8',
  });
}

test('prefers bun check and forwards project arguments', async (t) => {
  const root = await fixture(
    t,
    `if (process.argv.includes('--help')) process.exit(0);
console.log(JSON.stringify(process.argv.slice(2)));`
  );
  const result = run(root, ['-p', 'tsconfig.json']);
  assert.equal(result.status, 0, String(result.stderr));
  assert.deepEqual(JSON.parse(String(result.stdout)), [
    'check',
    '-p',
    'tsconfig.json',
  ]);
  assert.equal(result.stderr, '');
});

test('preserves Bun diagnostics and failure without falling back', async (t) => {
  const root = await fixture(
    t,
    `if (process.argv.includes('--help')) process.exit(0);
console.error('fixture type error');
process.exit(42);`
  );
  const result = run(root);
  assert.equal(result.status, 42);
  assert.equal(result.stderr, 'fixture type error\n');
});

test('fails when the selected checker is interrupted', async (t) => {
  const root = await fixture(
    t,
    `if (process.argv.includes('--help')) process.exit(0);
process.kill(process.pid, 'SIGTERM');`
  );
  assert.equal(run(root).status, 1);
});

for (const [name, bun] of [
  ['missing Bun', undefined],
  ['Bun without check', 'process.exit(1);'],
] as const) {
  test(`falls back to local tsc with ${name} without emitting`, async (t) => {
    const root = await fixture(t, bun);
    const result = run(root, ['-p', 'tsconfig.json', '--listEmittedFiles']);
    assert.equal(
      result.status,
      0,
      String(result.stdout) + String(result.stderr)
    );
    assert.match(String(result.stderr), /falling back to tsc/);
    assert.equal(result.stdout, '');
  });

  test(`preserves tsc diagnostics and failure with ${name}`, async (t) => {
    const root = await fixture(t, bun);
    await writeFile(
      join(root, 'index.ts'),
      'export const value: string = 123;\n'
    );
    const result = run(root);
    assert.equal(result.status, 2);
    assert.match(String(result.stdout), /TS2322/);
  });
}

test('explains how to recover when neither checker is available', async (t) => {
  const root = await fixture(t);
  const script = join(root, 'typecheck.ts');
  await copyFile(runner, script);
  const result = run(root, [], script);
  assert.equal(result.status, 1);
  assert.match(
    String(result.stderr),
    /neither `bun check` nor typescript is available; run npm ci/
  );
});

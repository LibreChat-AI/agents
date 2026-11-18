import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { cp, mkdtemp, writeFile, rm } from 'node:fs/promises';
import manifest from '../package.json' with { type: 'json' };
import { walkDeclarations } from './light-entries.mjs';

const run = promisify(execFile);
test('runtime declarations resolve outside the workspace', async () => {
  const declarations = await walkDeclarations('./runtime');
  assert.deepEqual(declarations.missing, []);
  assert.deepEqual(
    declarations.external.filter((name) => name.startsWith('@/')),
    []
  );
});
const root = path.resolve(fileURLToPath(import.meta.url), '../..');
test(
  'packed runtime works in an independent consumer with no LibreChat checkout',
  { timeout: 300000 },
  async () => {
    const directory = await mkdtemp(
      path.join(os.tmpdir(), 'agents-runtime-package-')
    );
    try {
      const { stdout } = await run(
        'npm',
        ['pack', '--ignore-scripts', '--json', '--pack-destination', directory],
        { cwd: root }
      );
      const [{ filename }] = JSON.parse(stdout);
      await writeFile(
        path.join(directory, 'package.json'),
        JSON.stringify({
          private: true,
          type: 'module',
          dependencies: {
            '@librechat/agents': `file:./${filename}`,
            '@langchain/core': manifest.dependencies['@langchain/core'],
            '@langchain/langgraph':
              manifest.dependencies['@langchain/langgraph'],
            zod: manifest.dependencies.zod,
          },
        })
      );
      await cp(path.join(root, 'test/runtime'), directory, { recursive: true });
      await run(
        'npm',
        [
          'install',
          '--ignore-scripts',
          '--omit=dev',
          '--no-audit',
          '--no-fund',
        ],
        { cwd: directory, timeout: 240000 }
      );
      await run(
        process.execPath,
        [
          path.join(root, 'node_modules/typescript/bin/tsc'),
          '--noEmit',
          '--strict',
          '--skipLibCheck',
          '--module',
          'ESNext',
          '--moduleResolution',
          'bundler',
          '--target',
          'ES2022',
          'consumer-types.mts',
        ],
        { cwd: directory, timeout: 30000 }
      );
      await run(
        process.execPath,
        [
          path.join(root, 'node_modules/typescript/bin/tsc'),
          '--noEmit',
          '--strict',
          '--skipLibCheck',
          '--module',
          'CommonJS',
          '--moduleResolution',
          'node',
          '--target',
          'ES2022',
          'consumer-types.cts',
        ],
        { cwd: directory, timeout: 30000 }
      );
      const { stdout: output } = await run(process.execPath, ['consumer.mjs'], {
        cwd: directory,
        timeout: 30000,
      });
      console.log(output.trim());
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }
);

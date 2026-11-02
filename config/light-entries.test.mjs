import path from 'node:path';
import test from 'node:test';
import { access } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  lightSubpaths,
  neutralSubpaths,
  probe,
  walkDeclarations,
} from './light-entries.mjs';
import { packageEntries } from './package-entries.mjs';
import manifest from '../package.json' with { type: 'json' };

const root = path.resolve(fileURLToPath(import.meta.url), '../..');
const run = promisify(execFile);

for (const format of ['esm', 'cjs']) {
  test(`schema conversion loads with only declared dependencies (${format})`, async () => {
    const extension = format === 'esm' ? 'mjs' : 'cjs';
    const script = `
      import assert from 'node:assert/strict';
      import { isBuiltin, registerHooks } from 'node:module';
      import manifest from './package.json' with { type: 'json' };
      import { z } from 'zod';
      const entry = new URL('./dist/${format}/utils/schema.${extension}', import.meta.url);
      registerHooks({
        resolve(specifier, context, nextResolve) {
          if (context.parentURL === entry.href && !isBuiltin(specifier) &&
              !specifier.startsWith('.') && !specifier.startsWith('file:')) {
            const name = specifier.split('/').slice(0, specifier.startsWith('@') ? 2 : 1).join('/');
            assert.ok(manifest.dependencies[name] || manifest.peerDependencies?.[name],
              'Undeclared runtime dependency: ' + name);
          }
          return nextResolve(specifier, context);
        },
      });
      const { toJsonSchema } = await import(entry.href);
      assert.equal(toJsonSchema(z.object({ query: z.string() })).properties.query.type, 'string');
    `;
    await run(process.execPath, ['--input-type=module', '--eval', script], {
      cwd: root,
    });
  });
}

test('light subpaths are built entries with esm, cjs and declaration outputs', async () => {
  for (const subpath of lightSubpaths) {
    const target = manifest.exports[subpath];
    assert.ok(target, `package.json exports ${subpath}`);
    const entry = target.import.replace(/^\.\/dist\/esm\/|\.mjs$/g, '');
    assert.equal(packageEntries[entry], `src/${entry}.ts`);
    assert.equal(target.require, `./dist/cjs/${entry}.cjs`);
    assert.equal(target.types, `./dist/types/${entry}.d.ts`);
    assert.deepEqual(manifest.typesVersions['*'][subpath.slice(2)], [
      target.types.slice(2),
    ]);
    for (const file of [target.types, target.import, target.require]) {
      await access(path.join(root, file));
    }
  }
});

for (const subpath of lightSubpaths) {
  for (const format of ['esm', 'cjs']) {
    test(`${subpath} loads without third-party packages (${format})`, async (t) => {
      const result = await probe(subpath, format);
      t.diagnostic(`${result.modules.length} modules`);
      assert.deepEqual(result.packages, []);
      assert.ok(result.modules.length > 0);
      assert.ok(
        result.modules.every((file) => file.startsWith(`dist/${format}/`)),
        result.modules.join('\n')
      );
    });
  }

  test(`${subpath} declarations resolve inside the package`, async (t) => {
    const result = await walkDeclarations(subpath);
    t.diagnostic(`${result.files.length} declaration files`);
    assert.deepEqual(result.external, []);
    assert.deepEqual(result.missing, []);
  });
}

for (const subpath of neutralSubpaths) {
  for (const format of ['esm', 'cjs']) {
    test(`${subpath} loads without Node built-ins (${format})`, async () => {
      const result = await probe(subpath, format);
      assert.deepEqual(result.builtins, []);
    });
  }
}

test('./decisions/node reaches only node:child_process', async () => {
  for (const format of ['esm', 'cjs']) {
    const result = await probe('./decisions/node', format);
    assert.deepEqual(result.builtins, ['child_process']);
  }
});

test('the probe reports third-party packages a subpath reaches', async () => {
  const result = await probe('./langchain/messages', 'esm');
  assert.ok(result.packages.includes('@langchain/core/messages'));
});

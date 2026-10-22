import path from 'node:path';
import test from 'node:test';
import { access } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

import { lightSubpaths, probe, walkDeclarations } from './light-entries.mjs';
import { packageEntries } from './package-entries.mjs';
import manifest from '../package.json' with { type: 'json' };

const root = path.resolve(fileURLToPath(import.meta.url), '../..');

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

test('the probe reports third-party packages a subpath reaches', async () => {
  const result = await probe('./langchain/messages', 'esm');
  assert.ok(result.packages.includes('@langchain/core/messages'));
});

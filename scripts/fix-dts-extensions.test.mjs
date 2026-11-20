import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fixDtsExtensionsInDir } from './fix-dts-extensions.mjs';

test('rewrites extensionless relative file and directory imports/exports in .d.ts files', () => {
  const root = mkdtempSync(join(tmpdir(), 'fix-dts-ext-'));
  try {
    mkdirSync(join(root, 'graphs'), { recursive: true });
    mkdirSync(join(root, 'tools'), { recursive: true });
    writeFileSync(join(root, 'run.d.ts'), 'export declare class Run {}\n');
    writeFileSync(join(root, 'graphs', 'index.d.ts'), 'export declare const g: number;\n');
    writeFileSync(join(root, 'tools', 'Calculator.d.ts'), 'export declare const calc: number;\n');
    writeFileSync(
      join(root, 'index.d.ts'),
      [
        "export * from './run';",
        "export * from './graphs';",
        "export { calc } from './tools/Calculator';",
        "import type { Run } from './run';",
        "export type Lazy = import('./tools/Calculator').default;",
        "export * from './run.js';",
        "import pkg from 'some-package';",
      ].join('\n') + '\n',
    );

    const updated = fixDtsExtensionsInDir(root);
    assert.equal(updated, 1);

    const result = readFileSync(join(root, 'index.d.ts'), 'utf8');
    assert.equal(
      result,
      [
        "export * from './run.js';",
        "export * from './graphs/index.js';",
        "export { calc } from './tools/Calculator.js';",
        "import type { Run } from './run.js';",
        "export type Lazy = import('./tools/Calculator.js').default;",
        "export * from './run.js';",
        "import pkg from 'some-package';",
      ].join('\n') + '\n',
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

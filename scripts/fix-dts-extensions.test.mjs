import test from 'node:test';
import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';

const script = fileURLToPath(
  new URL('./fix-dts-extensions.mjs', import.meta.url)
);

test('rewrites only relative module specifiers that have declaration targets', (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'agents-declarations-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  for (const name of [
    'file.d.ts',
    'file/index.d.ts',
    'barrel/index.d.ts',
    'provider.eager.d.ts',
    'already.js.d.ts',
  ]) {
    const file = join(directory, name);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, 'export {};\n');
  }

  const source = String.raw`export * from './file';
export type * from './barrel';
import type { Model } from './provider.eager';
import './barrel/';
import Legacy = require('./file');
type Ref = import('./file').Model;
import type { Escaped } from './\u0066ile';
declare module './file' {}
export * from './already.js';
export * from '@langchain/core/messages';
export * from './missing';
export declare const text = "from './file'";
/** @see import('./file').Model */
`;
  const expected = String.raw`export * from './file.js';
export type * from './barrel/index.js';
import type { Model } from './provider.eager.js';
import './barrel/index.js';
import Legacy = require('./file.js');
type Ref = import('./file.js').Model;
import type { Escaped } from './\u0066ile.js';
declare module './file.js' {}
export * from './already.js';
export * from '@langchain/core/messages';
export * from './missing';
export declare const text = "from './file'";
/** @see import('./file').Model */
`;
  const file = join(directory, 'index.d.ts');
  writeFileSync(file, source);

  for (let i = 0; i < 2; i++) {
    execFileSync(process.execPath, [script, directory]);
    assert.equal(readFileSync(file, 'utf8'), expected);
  }
});

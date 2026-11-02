import { tmpdir } from 'os';
import fs from 'fs/promises';
import { join, relative } from 'path';
import { AIMessage } from '@langchain/core/messages';
import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from 'fs';
import type { VerifierContext } from '../types';
import { resolveWorkspacePathSafe } from '@/tools/local/LocalExecutionEngine';
import { artifactExists, jsonFileValid } from '../verifiers';
import { createCompletionGate } from '../completionGate';
import { MemoryRunLedger } from '../ledger';

let directory: string;
let root: string;
let outside: string;

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), 'artifact-containment-'));
  root = join(directory, 'workspace');
  outside = join(directory, 'workspace-other');
  mkdirSync(root);
  mkdirSync(outside);
  mkdirSync(join(root, 'data'));
  for (const path of [
    join(root, 'ANSWER.json'),
    join(root, '..answer.json'),
    join(root, 'data', 'ANSWER.json'),
    join(outside, 'ANSWER.json'),
  ]) {
    writeFileSync(path, '{"answer":42}');
  }
  writeFileSync(join(root, 'bad.json'), '{');
  symlinkSync('../workspace-other/ANSWER.json', join(root, 'outside.json'));
  symlinkSync(outside, join(root, 'outside-directory'), 'dir');
  symlinkSync('outside.json', join(root, 'chain.json'));
  symlinkSync('ANSWER.json', join(root, 'inside.json'));
  symlinkSync('data', join(root, 'inside-directory'), 'dir');
  symlinkSync('missing.json', join(root, 'dangling.json'));
  symlinkSync('loop.json', join(root, 'loop.json'));
  symlinkSync(root, join(directory, 'workspace-alias'), 'dir');
});

afterEach(() => {
  jest.restoreAllMocks();
  rmSync(directory, { recursive: true, force: true });
});

function context(path: VerifierContext['params']['path']): VerifierContext {
  return {
    runId: 'containment',
    requirement: { mark: 'artifact', verifier: 'test' },
    params: { path },
    finalText: null,
    signal: new AbortController().signal,
  };
}

describe.each([
  ['artifact_exists', artifactExists],
  ['json_file_valid', jsonFileValid],
] as const)('%s containment', (_id, createVerifier) => {
  it.each(['outside.json', 'outside-directory/ANSWER.json', 'chain.json'])(
    'rejects escaped target %s before artifact I/O',
    async (path) => {
      const read = jest.spyOn(fs, 'readFile');
      const stat = jest.spyOn(fs, 'stat');
      expect((await createVerifier(root).run(context(path))).passed).toBe(false);
      expect(read).not.toHaveBeenCalled();
      expect(stat).not.toHaveBeenCalled();
    }
  );

  it.each([
    'ANSWER.json',
    '..answer.json',
    'inside.json',
    'inside-directory/ANSWER.json',
  ])('accepts existing in-workspace artifact %s', async (path) => {
    expect((await createVerifier(root).run(context(path))).passed).toBe(true);
  });

  it('resolves a symlinked workspace root and relative root', async () => {
    for (const configuredRoot of [
      join(directory, 'workspace-alias'),
      relative(process.cwd(), root),
    ]) {
      expect(
        (await createVerifier(configuredRoot).run(context('inside.json'))).passed
      ).toBe(true);
      expect(
        (await createVerifier(configuredRoot).run(context('outside.json'))).passed
      ).toBe(false);
    }
  });

  it.each(['missing.json', 'dangling.json', 'loop.json', 'data'])(
    'rejects unavailable file %s',
    async (path) => {
      expect((await createVerifier(root).run(context(path))).passed).toBe(false);
    }
  );

  it('keeps relative-only path validation', async () => {
    for (const path of [
      '',
      '.',
      '../workspace-other/ANSWER.json',
      join(root, 'ANSWER.json'),
      42,
      null,
    ]) {
      expect((await createVerifier(root).run(context(path))).passed).toBe(false);
    }
  });

  it('admits a completion mark only after the escaped artifact is replaced', async () => {
    const verifier = createVerifier(root);
    const ledger = new MemoryRunLedger();
    const gate = createCompletionGate({
      requirements: [{
        mark: 'artifact', verifier: verifier.id, params: { path: 'outside.json' },
      }],
      verifiers: [verifier],
      ledger,
    });
    const input = {
      hook_event_name: 'Stop' as const,
      runId: 'containment',
      messages: [new AIMessage('done')],
      stopHookActive: false,
      continuationCount: 0,
      continuationBudgetRemaining: 2,
    };
    expect((await gate.hook(input, new AbortController().signal)).decision).toBe('block');
    expect(gate.getCompletion()).toMatchObject({
      verified: false, marks: [], missing: ['artifact'], attempts: 1,
    });
    expect(ledger.rows().map((row) => row.kind)).toEqual(['verification_failed']);

    unlinkSync(join(root, 'outside.json'));
    writeFileSync(join(root, 'outside.json'), '{"answer":43}');
    expect(await gate.hook(
      { ...input, continuationCount: 1 }, new AbortController().signal
    )).toEqual({});
    expect(gate.getCompletion()).toMatchObject({
      verified: true, missing: [], attempts: 2,
      marks: [{ mark: 'artifact', verifier: verifier.id }],
    });
    expect(ledger.rows().map((row) => row.kind)).toEqual([
      'verification_failed', 'mark_admitted', 'run_completed',
    ]);
  });
});

it('still requires JSON syntax only for the JSON verifier', async () => {
  expect((await artifactExists(root).run(context('bad.json'))).passed).toBe(true);
  expect((await jsonFileValid(root).run(context('bad.json'))).passed).toBe(false);
});

it.each(['read', 'write'] as const)(
  'shared resolver accepts literal double-dot-prefixed names for %s without allowing traversal',
  async (intent) => {
    await expect(resolveWorkspacePathSafe('..answer.json', { cwd: root }, intent))
      .resolves.toBe(join(root, '..answer.json'));
    await expect(resolveWorkspacePathSafe('../workspace-other/ANSWER.json', { cwd: root }, intent))
      .rejects.toThrow('outside the local workspace');
  }
);

import { describe, it, expect } from '@jest/globals';
import type { PostToolBatchEntry, PostToolBatchHookInput } from '../types';
import {
  createStopOnPassingCheckHook,
  outputText,
} from '../createStopOnPassingCheckHook';

const entry = (
  toolName: string,
  toolInput: Record<string, unknown>,
  toolOutput: unknown = 'ok'
): PostToolBatchEntry => ({
  toolName,
  toolInput,
  toolUseId: `${toolName}-${Math.random()}`,
  status: 'success',
  toolOutput,
});
const batch = (...entries: PostToolBatchEntry[]): PostToolBatchHookInput =>
  ({
    hook_event_name: 'PostToolBatch',
    runId: 'r',
    entries,
  }) as unknown as PostToolBatchHookInput;
const edit = entry('edit_file', { path: 'a.py' });
const passingTest = entry(
  'bash_tool',
  { command: 'python3 -m pytest -q tests/test_a.py' },
  '3 passed'
);
const failingTest = entry(
  'bash_tool',
  { command: 'python3 -m pytest -q' },
  '1 failed\nexit_code: 1'
);

describe('createStopOnPassingCheckHook', () => {
  it('does not let a test verify an edit from the same concurrent batch', async () => {
    const hook = createStopOnPassingCheckHook();
    expect(await hook(batch(edit, passingTest), {} as never)).toEqual({});
    expect(await hook(batch(passingTest), {} as never)).toMatchObject({
      preventContinuation: true,
    });
  });

  it('keeps a failing check in a batch even when a sibling test passes', async () => {
    const hook = createStopOnPassingCheckHook();
    expect(await hook(batch(edit), {} as never)).toEqual({});
    expect(await hook(batch(failingTest, passingTest), {} as never)).toEqual(
      {}
    );
  });

  it('does not count a test that races a background edit in the same call', async () => {
    const hook = createStopOnPassingCheckHook();
    const racing = entry(
      'bash_tool',
      { command: 'sleep 5 && sed -i \'s/a/b/\' a.py & python3 -m pytest tests' },
      '3 passed'
    );
    const sequential = entry(
      'bash_tool',
      { command: 'sed -i \'s/a/b/\' a.py && python3 -m pytest tests 2>&1' },
      '3 passed'
    );
    expect(await hook(batch(racing), {} as never)).toEqual({});
    expect(await hook(batch(sequential), {} as never)).toMatchObject({
      preventContinuation: true,
    });
  });

  it('ignores edit commands that appear only as data', async () => {
    const hook = createStopOnPassingCheckHook();
    const search = entry('bash_tool', { command: 'rg \'sed -i\' scripts' });
    const longForm = entry('bash_tool', {
      command: 'sed --in-place \'s/a/b/\' a.py',
    });
    const fresh = createStopOnPassingCheckHook();
    expect(await fresh(batch(longForm), {} as never)).toEqual({});
    expect(await fresh(batch(passingTest), {} as never)).toMatchObject({
      preventContinuation: true,
    });
    expect(await hook(batch(search), {} as never)).toEqual({});
    expect(await hook(batch(passingTest), {} as never)).toEqual({});
  });

  it('does not treat a failed edit or a dry-run patch as a change', async () => {
    const hook = createStopOnPassingCheckHook();
    const failedEdit = { ...entry('edit_file', { path: 'a.py' }), status: 'error' };
    const dryRun = entry('bash_tool', {
      command: 'git apply --check fix.patch && python3 -m pytest -q',
    });
    expect(
      await hook(
        batch(failedEdit as PostToolBatchEntry, dryRun, passingTest),
        {} as never
      )
    ).toEqual({});
  });

  it('stops after a passing test that follows an edit, across batches', async () => {
    const hook = createStopOnPassingCheckHook();
    expect(await hook(batch(edit), {} as never)).toEqual({});
    expect(await hook(batch(passingTest), {} as never)).toMatchObject({
      preventContinuation: true,
    });
  });

  it('never stops without a code change, on a failing test, or when an edit follows the test', async () => {
    expect(
      await createStopOnPassingCheckHook()(batch(passingTest), {} as never)
    ).toEqual({});
    expect(
      await createStopOnPassingCheckHook()(
        batch(edit, failingTest),
        {} as never
      )
    ).toEqual({});
    expect(
      await createStopOnPassingCheckHook()(
        batch(edit, passingTest, edit),
        {} as never
      )
    ).toEqual({});
  });

  it('ignores non-test shell commands and errored test calls', async () => {
    const hook = createStopOnPassingCheckHook();
    expect(
      await hook(
        batch(edit, entry('bash_tool', { command: 'ls' })),
        {} as never
      )
    ).toEqual({});
    expect(
      await hook(
        batch({
          ...passingTest,
          status: 'error',
          toolOutput: undefined,
          error: 'boom',
        }),
        {} as never
      )
    ).toEqual({});
  });

  it('reads content-and-artifact tool outputs', () => {
    expect(outputText(['text', { a: 1 }])).toBe('text');
    expect(outputText({ content: 'c' })).toBe('c');
    expect(outputText(42)).toBe('');
  });
});

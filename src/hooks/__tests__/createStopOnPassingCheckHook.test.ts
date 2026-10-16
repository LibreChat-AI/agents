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

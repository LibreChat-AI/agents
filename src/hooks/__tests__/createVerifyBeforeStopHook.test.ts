import { describe, it, expect } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { StopHookInput } from '../types';
import {
  createVerifyBeforeStopHook,
  verifiedSinceLastChange,
} from '../createVerifyBeforeStopHook';

let seq = 0;
function call(
  name: string,
  args: Record<string, unknown>,
  output: string
): BaseMessage[] {
  const id = `call_${seq++}`;
  return [
    new AIMessage({
      content: '',
      tool_calls: [{ id, name, args, type: 'tool_call' }],
    }),
    new ToolMessage({ content: output, tool_call_id: id, name }),
  ];
}
const edit = (): BaseMessage[] =>
  call('edit_file', { path: 'a.py', old_text: 'a', new_text: 'b' }, 'ok');
const test = (output = 'stdout:\n3 passed\n'): BaseMessage[] =>
  call('bash_tool', { command: 'python3 -m pytest tests' }, output);

function stopInput(
  messages: BaseMessage[],
  continuationCount = 0
): StopHookInput {
  return {
    hook_event_name: 'Stop',
    runId: 'r',
    messages,
    stopHookActive: continuationCount > 0,
    continuationCount,
    continuationBudgetRemaining: 3,
  } as StopHookInput;
}

describe('verifiedSinceLastChange', () => {
  it('counts runner invocations but not commands that only mention a runner', () => {
    const run = (command: string): BaseMessage[] =>
      call('bash_tool', { command }, 'stdout:\nok\n');
    expect(
      verifiedSinceLastChange([...edit(), ...run('rg pytest pyproject.toml')])
    ).toBe(false);
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...run('cd repo && PYTHONPATH=src python3 -m pytest -x tests'),
      ])
    ).toBe(true);
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...run('rg \'pytest|jest\' package.json'),
      ])
    ).toBe(false);
  });

  it('handles redirections, listing modes and an edit followed by tests in one call', () => {
    const run = (command: string, output = 'stdout:\nok\n'): BaseMessage[] =>
      call('bash_tool', { command }, output);
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...run('python3 -m pytest tests 2>&1'),
      ])
    ).toBe(true);
    expect(
      verifiedSinceLastChange([...edit(), ...run('npx jest --listTests')])
    ).toBe(false);
    expect(
      verifiedSinceLastChange([...edit(), ...run('cargo test --no-run')])
    ).toBe(false);
    expect(
      verifiedSinceLastChange([...edit(), ...run('npx vitest list')])
    ).toBe(false);
    for (const command of [
      'go test -list . ./...',
      'ctest -N',
      'rspec --dry-run',
      'git diff --quiet || npm test',
      'true || (echo setup && python3 -m pytest)',
      'go test -list=. ./...',
      'python3 -m pytest tests >/tmp/tests.log 2>&1 &',
      'python3 -m pytest --setup-plan',
      'npx jest --showConfig',
      'python3 -m pytest --collectonly',
      '(python3 -m pytest --collect-only)',
      'ctest --show-only=json-v1',
      'if [ ! -d tests ]; then exit 0; fi; python3 -m pytest tests',
    ]) {
      expect(verifiedSinceLastChange([...edit(), ...run(command)])).toBe(false);
    }
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...run('python3 -m pytest --collect-only -q'),
      ])
    ).toBe(false);
    expect(
      verifiedSinceLastChange(
        run('sed -i \'s/a/b/\' a.py && python3 -m pytest tests')
      )
    ).toBe(true);
    expect(
      verifiedSinceLastChange(
        run('python3 -m pytest tests && sed -i \'s/a/b/\' a.py')
      )
    ).toBe(false);
  });

  it('does not let a test verify an edit made in the same assistant turn', () => {
    const editId = `call_${seq++}`;
    const testId = `call_${seq++}`;
    const turn: BaseMessage[] = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: editId,
            name: 'edit_file',
            args: { path: 'a.py' },
            type: 'tool_call',
          },
          {
            id: testId,
            name: 'bash_tool',
            args: { command: 'python3 -m pytest tests' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({ content: 'ok', tool_call_id: editId, name: 'edit_file' }),
      new ToolMessage({
        content: 'stdout:\n3 passed\n',
        tool_call_id: testId,
        name: 'bash_tool',
      }),
    ];
    expect(verifiedSinceLastChange(turn)).toBe(false);
    expect(verifiedSinceLastChange([...turn, ...test()])).toBe(true);
  });

  it('judges only the current user turn and treats negative exit codes as failures', () => {
    expect(
      verifiedSinceLastChange([...edit(), new HumanMessage('what changed?')])
    ).toBeUndefined();
    const meta = new HumanMessage({
      content: 'run the tests',
      additional_kwargs: { isMeta: true },
    });
    expect(verifiedSinceLastChange([...edit(), meta])).toBe(false);
    const hookContext = new HumanMessage({
      content: 'lint passed',
      additional_kwargs: { role: 'system', source: 'hook' },
    });
    expect(verifiedSinceLastChange([...edit(), hookContext])).toBe(false);
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...test('stdout:\nkilled\nexit_code: -9\n'),
      ])
    ).toBe(false);
  });

  it('ignores trailing newlines and semicolons after a test run', () => {
    for (const command of ['python3 -m pytest -q\n', 'python3 -m pytest -q;']) {
      expect(
        verifiedSinceLastChange([
          ...edit(),
          ...call('bash_tool', { command }, 'stdout:\n3 passed\n'),
        ])
      ).toBe(true);
    }
  });

  it('ignores test runs whose exit code another command replaces', () => {
    for (const command of [
      'python3 -m pytest tests; echo done',
      'python3 -m pytest tests || true',
      'python3 -m pytest tests 2>&1 | tail -20',
    ]) {
      expect(
        verifiedSinceLastChange([
          ...edit(),
          ...call('bash_tool', { command }, 'stdout:\n1 failed\n'),
        ])
      ).toBe(false);
    }
  });

  it('gives the same answer on every call for a global custom pattern', () => {
    const messages = [...edit(), ...test()];
    const config = { testCommandPattern: /pytest/g };
    expect(verifiedSinceLastChange(messages, config)).toBe(true);
    expect(verifiedSinceLastChange(messages, config)).toBe(true);
  });

  it('does not count an errored test tool result as verification', () => {
    const id = `call_${seq++}`;
    const errored: BaseMessage[] = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id,
            name: 'bash_tool',
            args: { command: 'python3 -m pytest tests' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'Error: failed to spawn process',
        tool_call_id: id,
        name: 'bash_tool',
        status: 'error',
      }),
    ];
    expect(verifiedSinceLastChange([...edit(), ...errored])).toBe(false);
  });

  it('is undefined when nothing changed', () => {
    expect(
      verifiedSinceLastChange([new HumanMessage('hi'), ...test()])
    ).toBeUndefined();
  });

  it('requires a passing test after the last change', () => {
    expect(verifiedSinceLastChange([...edit()])).toBe(false);
    expect(verifiedSinceLastChange([...edit(), ...test()])).toBe(true);
    expect(verifiedSinceLastChange([...edit(), ...test(), ...edit()])).toBe(
      false
    );
  });

  it('treats a non-zero exit code or timeout as not verified', () => {
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...test('stdout:\n1 failed\nexit_code: 1\n'),
      ])
    ).toBe(false);
    expect(
      verifiedSinceLastChange([
        ...edit(),
        ...test('stdout:\n\ntimed_out: true\n'),
      ])
    ).toBe(false);
  });

  it('counts in-place shell edits as changes and ignores non-test commands', () => {
    const sed = call(
      'bash_tool',
      { command: 'sed -i s/a/b/ a.py' },
      'stdout: Empty.'
    );
    const cat = call('bash_tool', { command: 'cat a.py' }, 'stdout:\nb\n');
    expect(verifiedSinceLastChange([...sed, ...cat])).toBe(false);
    expect(verifiedSinceLastChange([...sed, ...test()])).toBe(true);
  });
});

describe('createVerifyBeforeStopHook', () => {
  it('blocks once with guidance when the last change is unverified', () => {
    const hook = createVerifyBeforeStopHook();
    const signal = new AbortController().signal;
    const first = hook(stopInput([...edit()]), signal);
    expect(first).toMatchObject({ decision: 'block' });
    expect((first as { additionalContext?: string }).additionalContext).toMatch(
      /tests/
    );
    expect(hook(stopInput([...edit()], 1), signal)).toEqual({});
  });

  it('does not block when the run has no continuation budget left', () => {
    const hook = createVerifyBeforeStopHook({ maxContinuations: 2 });
    const signal = new AbortController().signal;
    const input = {
      ...stopInput([...edit()]),
      continuationBudgetRemaining: 0,
    } as StopHookInput;
    expect(hook(input, signal)).toEqual({});
  });

  it('lets verified and unchanged runs finish', () => {
    const hook = createVerifyBeforeStopHook();
    const signal = new AbortController().signal;
    expect(hook(stopInput([...edit(), ...test()]), signal)).toEqual({});
    expect(hook(stopInput([new HumanMessage('question')]), signal)).toEqual({});
  });
});

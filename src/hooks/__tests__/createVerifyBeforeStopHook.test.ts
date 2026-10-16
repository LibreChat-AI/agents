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

  it('lets verified and unchanged runs finish', () => {
    const hook = createVerifyBeforeStopHook();
    const signal = new AbortController().signal;
    expect(hook(stopInput([...edit(), ...test()]), signal)).toEqual({});
    expect(hook(stopInput([new HumanMessage('question')]), signal)).toEqual({});
  });
});

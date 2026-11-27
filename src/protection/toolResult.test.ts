import { Command } from '@langchain/langgraph';
import { ToolMessage } from '@langchain/core/messages';
import type {
  ToolResultProtection,
  ToolResultProtectionResult,
} from './toolResult';
import {
  ToolResultProtectionError,
  inspectToolExecuteResult,
  createToolResultGate,
  inspectToolOutput,
  inspectToolText,
} from './toolResult';
import { ProviderTextProtectionError } from './providerText';

const RAW = 'reach alice@example.com';
const SAFE = 'reach [EMAIL_1]';
const call = { name: 'lookup', id: 'call-1' };

function approve(content: string): ToolResultProtectionResult {
  return {
    version: 1,
    ok: true,
    value: { content, replacements: 1, categories: [] },
  };
}

function policy(
  overrides: Partial<ToolResultProtection> = {}
): ToolResultProtection {
  return {
    version: 1,
    toolNames: ['lookup'],
    timeoutMs: 5000,
    maxAttemptBytes: 65536,
    maxBufferedBytes: 262144,
    classify: () => 'prose',
    inspect: ({ content }) =>
      approve(content.replace('alice@example.com', '[EMAIL_1]')),
    ...overrides,
  };
}

async function failure(work: Promise<unknown>): Promise<Error> {
  return work.then(
    () => new Error('expected a rejection'),
    (error: Error) => error
  );
}

describe('createToolResultGate', () => {
  it('returns one gate per policy so graphs share its budget', () => {
    const shared = policy();
    expect(createToolResultGate(shared)).toBe(createToolResultGate(shared));
    expect(createToolResultGate(shared).toolNames.has('lookup')).toBe(true);
  });

  it.each([
    ['an unknown version', { version: 2 as unknown as 1 }, 'incompatible'],
    ['no selected tools', { toolNames: [] }, 'incompatible'],
    ['a blank tool name', { toolNames: [' '] }, 'incompatible'],
    ['a zero timeout', { timeoutMs: 0 }, 'incompatible'],
    [
      'an attempt budget above the shared budget',
      { maxAttemptBytes: 10, maxBufferedBytes: 5 },
      'incompatible',
    ],
    [
      'a missing inspector',
      { inspect: undefined as unknown as ToolResultProtection['inspect'] },
      'unavailable',
    ],
    [
      'a missing classifier',
      { classify: undefined as unknown as ToolResultProtection['classify'] },
      'unavailable',
    ],
  ])('rejects %s', (_label, overrides, code) => {
    const error = ((): unknown => {
      try {
        return createToolResultGate(policy(overrides));
      } catch (caught) {
        return caught;
      }
    })();
    expect(error).toBeInstanceOf(ToolResultProtectionError);
    expect(error).toMatchObject({ code });
  });
});

describe('inspectToolText', () => {
  it('hands the inspector the call identity and returns its text', async () => {
    const inputs: Parameters<ToolResultProtection['inspect']>[0][] = [];
    const gate = createToolResultGate(
      policy({
        inspect: (input) => {
          inputs.push(input);
          return approve(SAFE);
        },
      })
    );
    await expect(inspectToolText(gate, call, RAW, 'error')).resolves.toBe(SAFE);
    expect(inputs).toHaveLength(1);
    expect(inputs[0]).toMatchObject({
      version: 1,
      content: RAW,
      toolName: 'lookup',
      toolCallId: 'call-1',
      target: {
        source: 'tool_argument',
        field: 'output',
        provenance: 'tool',
        outcome: 'error',
      },
    });
    expect(inputs[0].signal.aborted).toBe(false);
  });

  it('returns empty text without calling the inspector', async () => {
    const inspect = jest.fn(() => approve('unused'));
    const gate = createToolResultGate(policy({ inspect }));
    await expect(inspectToolText(gate, call, '', 'success')).resolves.toBe('');
    expect(inspect).not.toHaveBeenCalled();
  });

  it('fails with the code the inspector returned', async () => {
    const gate = createToolResultGate(
      policy({
        inspect: () => ({ version: 1, ok: false, error: { code: 'blocked' } }),
      })
    );
    const error = await failure(inspectToolText(gate, call, RAW, 'success'));
    expect(error).toBeInstanceOf(ToolResultProtectionError);
    expect(error).toBeInstanceOf(ProviderTextProtectionError);
    expect(error).toMatchObject({
      code: 'blocked',
      name: 'ToolResultProtectionError',
      message: 'Tool result protection: blocked',
    });
  });

  it.each([
    [
      'throws',
      (): ToolResultProtectionResult => {
        throw new Error(`inspector saw ${RAW}`);
      },
    ],
    [
      'rejects',
      (): Promise<ToolResultProtectionResult> =>
        Promise.reject(new Error(`inspector saw ${RAW}`)),
    ],
  ])(
    'reports an inspector that %s as unavailable without its message',
    async (_label, inspect) => {
      const gate = createToolResultGate(policy({ inspect }));
      const error = await failure(inspectToolText(gate, call, RAW, 'success'));
      expect(error).toMatchObject({ code: 'unavailable' });
      expect(error.message).not.toContain('alice');
      expect(JSON.stringify(error)).not.toContain('alice');
    }
  );

  it('fails as unsupported when the classifier rejects the text', async () => {
    const inspect = jest.fn(() => approve(SAFE));
    const gate = createToolResultGate(
      policy({ classify: () => 'unsupported', inspect })
    );
    const error = await failure(inspectToolText(gate, call, RAW, 'success'));
    expect(error).toMatchObject({ code: 'unsupported' });
    expect(inspect).not.toHaveBeenCalled();
  });

  it.each([
    ['a missing result', undefined],
    ['an unknown result version', { version: 2, ok: true }],
    [
      'non-string content',
      { version: 1, ok: true, value: { content: 7, replacements: 0 } },
    ],
  ])('fails as incompatible for %s', async (_label, result) => {
    const gate = createToolResultGate(
      policy({
        inspect: () => result as unknown as ToolResultProtectionResult,
      })
    );
    const error = await failure(inspectToolText(gate, call, RAW, 'success'));
    expect(error).toMatchObject({ code: 'incompatible' });
  });

  it('times out an inspector that never settles and aborts its signal', async () => {
    let signal: AbortSignal | undefined;
    const gate = createToolResultGate(
      policy({
        timeoutMs: 20,
        inspect: (input) => {
          signal = input.signal;
          return new Promise<ToolResultProtectionResult>(() => {});
        },
      })
    );
    const error = await failure(inspectToolText(gate, call, RAW, 'success'));
    expect(error).toMatchObject({ code: 'timeout' });
    expect(signal?.aborted).toBe(true);
  });

  it('fails as cancelled when the run aborts during inspection', async () => {
    const controller = new AbortController();
    const gate = createToolResultGate(
      policy({
        inspect: () => {
          controller.abort();
          return new Promise<ToolResultProtectionResult>(() => {});
        },
      })
    );
    const error = await failure(
      inspectToolText(gate, call, RAW, 'success', controller.signal)
    );
    expect(error).toMatchObject({ code: 'cancelled' });
  });

  it('fails as cancelled without inspecting when the run already aborted', async () => {
    const inspect = jest.fn(() => approve(SAFE));
    const gate = createToolResultGate(policy({ inspect }));
    const error = await failure(
      inspectToolText(gate, call, RAW, 'success', AbortSignal.abort())
    );
    expect(error).toMatchObject({ code: 'cancelled' });
    expect(inspect).not.toHaveBeenCalled();
  });

  it('fails as overflow past the attempt budget and frees the budget afterwards', async () => {
    const inspect = jest.fn(({ content }: { content: string }) =>
      approve(content)
    );
    const gate = createToolResultGate(
      policy({ maxAttemptBytes: 4096, maxBufferedBytes: 4096, inspect })
    );
    const error = await failure(
      inspectToolText(gate, call, 'x'.repeat(4096), 'success')
    );
    expect(error).toMatchObject({ code: 'overflow' });
    expect(inspect).not.toHaveBeenCalled();
    for (let round = 0; round < 3; round++) {
      await expect(
        inspectToolText(gate, call, 'x'.repeat(256), 'success')
      ).resolves.toBe('x'.repeat(256));
    }
  });

  it('fails as overflow when the inspector returns more than the budget allows', async () => {
    const gate = createToolResultGate(
      policy({
        maxAttemptBytes: 4096,
        maxBufferedBytes: 4096,
        inspect: () => approve('y'.repeat(4096)),
      })
    );
    const error = await failure(inspectToolText(gate, call, RAW, 'success'));
    expect(error).toMatchObject({ code: 'overflow' });
  });
});

describe('inspectToolOutput', () => {
  const gate = createToolResultGate(policy());

  it('inspects a bare string', async () => {
    await expect(inspectToolOutput(gate, call, RAW)).resolves.toBe(SAFE);
  });

  it('replaces message content and keeps the message identity fields', async () => {
    const message = new ToolMessage({
      id: 'message-1',
      name: 'lookup',
      tool_call_id: 'call-1',
      status: 'error',
      content: RAW,
      additional_kwargs: { kept: true },
    });
    const outcomes: string[] = [];
    const recording = createToolResultGate(
      policy({
        inspect: ({ target }) => {
          outcomes.push(target.outcome);
          return approve(SAFE);
        },
      })
    );
    const inspected = await inspectToolOutput(recording, call, message);
    expect(inspected).toBeInstanceOf(ToolMessage);
    expect(inspected).toMatchObject({
      id: 'message-1',
      name: 'lookup',
      tool_call_id: 'call-1',
      status: 'error',
      content: SAFE,
      additional_kwargs: { kept: true },
    });
    expect(outcomes).toEqual(['error']);
  });

  it('returns the same message when inspection changes nothing', async () => {
    const message = new ToolMessage({ tool_call_id: 'call-1', content: 'ok' });
    await expect(inspectToolOutput(gate, call, message)).resolves.toBe(message);
  });

  it.each([
    ['a Command', new Command({ update: { messages: [] } })],
    ['a plain object', { text: RAW }],
    [
      'a message carrying an artifact',
      new ToolMessage({ tool_call_id: 'call-1', content: RAW, artifact: {} }),
    ],
    [
      'a message with content blocks',
      new ToolMessage({
        tool_call_id: 'call-1',
        content: [{ type: 'text', text: RAW }],
      }),
    ],
  ])('fails as unsupported for %s', async (_label, output) => {
    const error = await failure(inspectToolOutput(gate, call, output));
    expect(error).toMatchObject({ code: 'unsupported' });
  });
});

describe('inspectToolExecuteResult', () => {
  const gate = createToolResultGate(policy());

  it('replaces success content and keeps the other fields', async () => {
    await expect(
      inspectToolExecuteResult(gate, call, {
        toolCallId: 'call-1',
        status: 'success',
        content: RAW,
        received_at: 12,
        outcome: 'Looked up',
      })
    ).resolves.toEqual({
      toolCallId: 'call-1',
      status: 'success',
      content: SAFE,
      received_at: 12,
      outcome: 'Looked up',
    });
  });

  it('replaces the error message and drops uninspected error content', async () => {
    await expect(
      inspectToolExecuteResult(gate, call, {
        toolCallId: 'call-1',
        status: 'error',
        content: RAW,
        errorMessage: RAW,
      })
    ).resolves.toEqual({
      toolCallId: 'call-1',
      status: 'error',
      content: '',
      errorMessage: SAFE,
    });
  });

  it('drops error content when the host gave no error message', async () => {
    await expect(
      inspectToolExecuteResult(gate, call, {
        toolCallId: 'call-1',
        status: 'error',
        content: RAW,
      })
    ).resolves.toEqual({ toolCallId: 'call-1', status: 'error', content: '' });
  });

  it.each([
    ['an artifact', { content: RAW, artifact: { files: [] } }],
    ['content blocks', { content: [{ type: 'text', text: RAW }] }],
  ])('fails as unsupported for %s', async (_label, fields) => {
    const error = await failure(
      inspectToolExecuteResult(gate, call, {
        toolCallId: 'call-1',
        status: 'success',
        ...fields,
      })
    );
    expect(error).toMatchObject({ code: 'unsupported' });
  });
});

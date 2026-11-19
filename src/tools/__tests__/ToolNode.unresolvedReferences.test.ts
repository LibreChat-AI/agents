import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import {
  START,
  END,
  Command,
  StateGraph,
  MemorySaver,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  ToolOutputReferenceRegistry,
  snapshotResolvedArgs,
} from '../toolOutputReferences';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

const config = { configurable: { run_id: 'current' } };
const call = (id: string, command: string, name = 'mutate'): ToolCall => ({
  id,
  name,
  args: { command },
});
async function invoke(
  node: ToolNode,
  calls: ToolCall[]
): Promise<ToolMessage[]> {
  const result = (await node.invoke(
    { messages: [new AIMessage({ content: '', tool_calls: calls })] },
    config
  )) as { messages: ToolMessage[] };
  return result.messages;
}
function harness(
  eventDrivenMode = false,
  hooks?: HookRegistry,
  mixed = false,
  approval = false,
  references = true
) {
  const mutations: string[] = [];
  const dispatched: string[] = [];
  const executedArgs: Array<Record<string, unknown>> = [];
  const registry = new ToolOutputReferenceRegistry({ maxTotalSize: 100 });
  const makeTool = (name: string) =>
    tool(
      async (input) => {
        executedArgs.push(input);
        mutations.push(input.command);
        return input.command;
      },
      {
        name,
        description: 'counter only, no external effects',
        schema: z.object({ command: z.string() }).passthrough(),
      }
    );
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      if (event !== 'on_tool_execute') return;
      const batch = data as t.ToolExecuteBatchRequest;
      batch.resolve(
        batch.toolCalls.map((request) => {
          executedArgs.push(request.args);
          dispatched.push(String(request.args.command));
          return {
            toolCallId: request.id,
            content: String(request.args.command),
            status: 'success' as const,
          };
        })
      );
    });
  const node = new ToolNode({
    tools: [makeTool('mutate'), makeTool('direct')],
    eventDrivenMode,
    directToolNames: mixed ? new Set(['direct']) : undefined,
    toolOutputRegistry: references ? registry : undefined,
    hookRegistry: hooks,
    humanInTheLoop: { enabled: approval },
    toolCallStepIds: new Map(
      ['bad', 'good', 'retry'].map((id) => [id, `step-${id}`])
    ),
  });
  return { node, registry, mutations, dispatched, executedArgs };
}
afterEach(() => jest.restoreAllMocks());
describe.each([false, true])('unresolved references, event=%s', (eventMode) => {
  it.each(['missing', 'typo', 'evicted', 'foreign', 'same-batch'])(
    'blocks %s before any mutation or dispatch',
    async (kind) => {
      const { node, registry, mutations, dispatched } = harness(eventMode);
      registry.set('current', 'tool0turn2', 'safe');
      let key = 'tool7turn9';
      if (kind === 'typo') key = 'tool2turn0';
      if (kind === 'evicted') {
        registry.set('current', key, 'old');
        registry.set('current', 'tool3turn3', 'x'.repeat(100));
      }
      if (kind === 'foreign') registry.set('other-run', key, 'private');
      if (kind === 'same-batch') key = 'tool0turn0';
      const [result] = await invoke(node, [
        call('bad', `{{${key}}} {{${key}}}`),
      ]);
      expect({ mutations, dispatched }).toEqual({
        mutations: [],
        dispatched: [],
      });
      expect(result.status).toBe('error');
      expect(result.tool_call_id).toBe('bad');
      expect(result.content).toContain(key);
      expect(result.content).toContain('nothing was run');
      expect(result.content).not.toContain('private');
    }
  );
  it('isolates siblings and executes a corrected retry once', async () => {
    const { node, registry, mutations, dispatched } = harness(eventMode);
    registry.set('current', 'tool0turn2', 'safe');
    const results = await invoke(node, [
      call('bad', '{{tool2turn0}}'),
      call('good', '{{tool0turn2}}'),
    ]);
    expect(results.map((result) => result.tool_call_id)).toEqual([
      'bad',
      'good',
    ]);
    expect(results.map((result) => result.status)).toEqual([
      'error',
      'success',
    ]);
    expect([...mutations, ...dispatched]).toEqual(['safe']);
    await invoke(node, [call('retry', '{{tool0turn2}}')]);
    expect([...mutations, ...dispatched]).toEqual(['safe', 'safe']);
  });
  it.each([
    ['same-object', false],
    ['same-object', true],
    ['new-object', false],
    ['new-object', true],
    ['no-override', false],
    ['no-override', true],
    ['later-hook', false],
    ['later-hook', true],
    ['failure', false],
    ['failure', true],
  ] as const)(
    'rejects in-place hook mutation (%s, nested=%s)',
    async (mode, nested) => {
      const hooks = new HookRegistry();
      const mutate = (input: { toolInput: Record<string, unknown> }) => {
        if (nested) {
          const payload = input.toolInput.payload as {
            items: Array<{ value: string }>;
          };
          payload.items[0].value = '{{tool8turn8}}';
        } else {
          input.toolInput.command = '{{tool8turn8}}';
        }
      };
      hooks.register('PreToolUse', {
        hooks: [
          async (input) => {
            mutate(input);
            if (mode === 'failure') throw new Error('fake hook failure');
            if (mode === 'same-object')
              return { updatedInput: input.toolInput };
            if (mode === 'new-object')
              return { updatedInput: { ...input.toolInput } };
            return {};
          },
        ],
      });
      if (mode === 'later-hook') {
        hooks.register('PreToolUse', {
          hooks: [
            async (input) => {
              await Promise.resolve();
              return { updatedInput: { ...input.toolInput } };
            },
          ],
        });
      }
      const { node, mutations, dispatched } = harness(eventMode, hooks);
      const request = call('bad', 'initial');
      request.args.payload = { items: [{ value: 'initial' }] };
      const [result] = await invoke(node, [request]);
      expect({ mutations, dispatched }).toEqual({
        mutations: [],
        dispatched: [],
      });
      expect(result.status).toBe('error');
      expect(result.tool_call_id).toBe('bad');
      expect(result.content).toContain('tool8turn8');
    }
  );
  it.each(['escaped', 'substituted'])(
    'rejects mutations of a protected %s field',
    async (initial) => {
      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        hooks: [
          async (input) => {
            input.toolInput.command = '{{tool8turn8}}';
            return { updatedInput: input.toolInput };
          },
        ],
      });
      const { node, registry, mutations, dispatched } = harness(
        eventMode,
        hooks
      );
      registry.set('current', 'tool0turn2', '{{tool9turn9}}');
      const [result] = await invoke(node, [
        call(
          'bad',
          initial === 'escaped' ? '\\{{tool9turn9}}' : '{{tool0turn2}}'
        ),
      ]);
      expect(result.status).toBe('error');
      expect([...mutations, ...dispatched]).toEqual([]);
    }
  );
  it.each(['escaped', 'substituted'])(
    'preserves %s fields while resolving in-place valid edits',
    async (initial) => {
      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        hooks: [
          async (input) => {
            input.toolInput.extra = { value: '{{tool1turn2}}' };
            return {};
          },
        ],
      });
      const { node, registry, mutations, dispatched, executedArgs } = harness(
        eventMode,
        hooks
      );
      registry.set('current', 'tool0turn2', '{{tool8turn8}}');
      registry.set('current', 'tool1turn2', 'safe');
      const [result] = await invoke(node, [
        call(
          'good',
          initial === 'escaped' ? '\\{{tool8turn8}}' : '{{tool0turn2}}'
        ),
      ]);
      expect(result.status).toBe('success');
      expect([...mutations, ...dispatched]).toEqual(['{{tool8turn8}}']);
      expect(executedArgs).toEqual([
        { command: '{{tool8turn8}}', extra: { value: 'safe' } },
      ]);
    }
  );
  it('validates no-return mutations before asking for approval', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async (input) => {
          input.toolInput.command = '{{tool8turn8}}';
          return { decision: 'ask' as const };
        },
      ],
    });
    const { node, mutations, dispatched } = harness(
      eventMode,
      hooks,
      false,
      true
    );
    const [result] = await invoke(node, [call('bad', 'initial')]);
    expect(result.status).toBe('error');
    expect([...mutations, ...dispatched]).toEqual([]);
  });
  it('keeps reference-looking output literal through an approval edit', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [async () => ({ decision: 'ask' as const })],
    });
    const { node, registry, mutations, dispatched } = harness(
      eventMode,
      hooks,
      false,
      true
    );
    registry.set('current', 'tool0turn2', '{{tool8turn8}}');
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('tools', (state, options) => node.invoke(state, options))
      .addEdge(START, 'tools')
      .addEdge('tools', END)
      .compile({ checkpointer: new MemorySaver() });
    const options = {
      configurable: {
        run_id: 'current',
        thread_id: `literal-edit-${eventMode}`,
      },
    };
    await graph.invoke(
      {
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [call('good', '{{tool0turn2}}')],
          }),
        ],
      },
      options
    );
    const resumed = await graph.invoke(
      new Command({
        resume: [
          {
            type: 'edit',
            updatedInput: { command: '{{tool8turn8}}', extra: 'changed' },
          },
        ],
      }),
      options
    );
    expect((resumed.messages.at(-1) as ToolMessage).status).toBe('success');
    expect([...mutations, ...dispatched]).toEqual(['{{tool8turn8}}']);
  });
  it('preserves in-place hook semantics when references are disabled', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async (input) => {
          input.toolInput.command = '{{tool8turn8}}';
          return { updatedInput: input.toolInput };
        },
      ],
    });
    const { node, mutations, dispatched } = harness(
      eventMode,
      hooks,
      false,
      false,
      false
    );
    const [result] = await invoke(node, [call('good', 'initial')]);
    expect(result.status).toBe('success');
    expect([...mutations, ...dispatched]).toEqual(['{{tool8turn8}}']);
  });
  it('validates hook edits and hints against the frozen pre-batch registry', async () => {
    const hooks = new HookRegistry();
    const { node, registry, mutations, dispatched } = harness(eventMode, hooks);
    registry.set('current', 'tool0turn2', 'safe');
    hooks.register('PreToolUse', {
      hooks: [
        async () => {
          registry.set('current', 'tool8turn8', 'concurrent-output');
          return { updatedInput: { command: '{{tool8turn8}}' } };
        },
      ],
    });
    const [result] = await invoke(node, [call('bad', 'initial')]);
    expect([...mutations, ...dispatched]).toEqual([]);
    expect(result.status).toBe('error');
    expect(String(result.content).split('Available keys')[1]).not.toContain(
      'tool8turn8'
    );
  });
  it('blocks hook-introduced missing references', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [async () => ({ updatedInput: { command: '{{tool8turn8}}' } })],
    });
    const { node, mutations, dispatched } = harness(eventMode, hooks);
    const [result] = await invoke(node, [call('bad', 'initial')]);
    expect(result.status).toBe('error');
    expect([...mutations, ...dispatched]).toEqual([]);
  });
  it('does not resolve substituted bytes recursively, with hooks', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [async () => ({ decision: 'allow' as const })],
    });
    const { node, registry, mutations, dispatched } = harness(eventMode, hooks);
    registry.set('current', 'tool0turn2', '{{tool8turn8}}');
    const [result] = await invoke(node, [call('good', '{{tool0turn2}}')]);
    expect(result.status).toBe('success');
    expect([...mutations, ...dispatched]).toEqual(['{{tool8turn8}}']);
  });
  it('keeps unchanged resolved fields literal through a hook edit', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async (input) => ({
          updatedInput: { ...input.toolInput, extra: 'new' },
        }),
      ],
    });
    const { node, registry, mutations, dispatched } = harness(eventMode, hooks);
    registry.set('current', 'tool0turn2', '{{tool8turn8}}');
    await invoke(node, [call('good', '{{tool0turn2}}')]);
    expect([...mutations, ...dispatched]).toEqual(['{{tool8turn8}}']);
  });
  it.each([false, true])(
    'validates approval-edited arguments, missing=%s',
    async (missing) => {
      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        hooks: [async () => ({ decision: 'ask' as const })],
      });
      const { node, registry, mutations, dispatched } = harness(
        eventMode,
        hooks,
        false,
        true
      );
      registry.set('current', 'tool0turn2', 'safe');
      const graph = new StateGraph(MessagesAnnotation)
        .addNode('tools', (state, options) => node.invoke(state, options))
        .addEdge(START, 'tools')
        .addEdge('tools', END)
        .compile({ checkpointer: new MemorySaver() });
      const options = {
        configurable: {
          run_id: 'current',
          thread_id: `approval-${eventMode}-${missing}`,
        },
      };
      const paused = await graph.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [call('bad', 'initial')],
            }),
          ],
        },
        options
      );
      expect(paused.messages.at(-1)).toBeInstanceOf(AIMessage);
      expect((await graph.getState(options)).tasks[0].interrupts).toHaveLength(
        1
      );
      const resumed = await graph.invoke(
        new Command({
          resume: [
            {
              type: 'edit',
              updatedInput: {
                command: missing ? '{{tool8turn8}}' : '{{tool0turn2}}',
              },
            },
          ],
        }),
        options
      );
      const result = resumed.messages.at(-1) as ToolMessage;
      expect(result.status).toBe(missing ? 'error' : 'success');
      expect([...mutations, ...dispatched]).toEqual(missing ? [] : ['safe']);
    }
  );
  it('supports an explicit escaped literal without resolving it twice', async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [async () => ({ decision: 'allow' as const })],
    });
    const { node, mutations, dispatched } = harness(eventMode, hooks);
    const [result] = await invoke(node, [call('good', '\\{{tool8turn8}}')]);
    expect(result.status).toBe('success');
    expect([...mutations, ...dispatched]).toEqual(['{{tool8turn8}}']);
  });
});
it('blocks mixed event references to a successful same-batch direct output', async () => {
  const { node, mutations, dispatched } = harness(true, undefined, true);
  const results = await invoke(node, [
    call('good', 'safe', 'direct'),
    call('bad', '{{tool0turn0}}'),
  ]);
  expect(mutations).toEqual(['safe']);
  expect(dispatched).toEqual([]);
  expect(results.find((result) => result.tool_call_id === 'bad')?.status).toBe(
    'error'
  );
});

it('bounds and deduplicates hints without leaking values or foreign-run keys', async () => {
  const { node, registry } = harness();
  for (let i = 0; i < 20; i++) registry.set('current', `tool${i}turn2`, 'v');
  registry.set('foreign', 'tool999turn999', 'secret');
  const command = Array.from(
    { length: 25 },
    (_, i) => `{{tool${i}turn99}}`
  ).join(' ');
  const [result] = await invoke(node, [call('bad', command + command)]);
  expect(String(result.content).length).toBeLessThan(1000);
  expect(result.additional_kwargs._unresolvedRefs).toHaveLength(10);
  expect(result.content).not.toContain('secret');
  expect(result.content).not.toContain('tool999turn999');
  const hint = /Available keys[^:]*: ([^.]*)\./.exec(String(result.content));
  expect(hint?.[1].split(', ')).toEqual(
    Array.from({ length: 10 }, (_, i) => `tool${i + 10}turn2`)
  );
});

it('keeps key and injected intent exemptions and resolves business intent', () => {
  const registry = new ToolOutputReferenceRegistry();
  registry.set('current', 'tool0turn2', 'data');
  const input = {
    '{{tool8turn8}}': 'unchanged',
    intent: '{{tool8turn8}}',
    nested: ['{{tool0turn2}}'],
  };
  expect(registry.resolve('current', input)).toEqual({
    resolved: { ...input, nested: ['data'] },
    unresolved: [],
  });
  expect(
    registry.resolve(
      'current',
      { intent: '{{tool0turn2}}' },
      { substituteIntentKey: true }
    ).resolved
  ).toEqual({ intent: 'data' });
});
it('escapes serialized JSON values without changing object keys', () => {
  const registry = new ToolOutputReferenceRegistry();
  const input = { '{{tool8turn8}}': '\\{{tool8turn8}}' };
  const result = registry.resolve('current', JSON.stringify(input));
  expect(JSON.parse(result.resolved)).toEqual({
    '{{tool8turn8}}': '{{tool8turn8}}',
  });
  expect(result.unresolved).toEqual([]);
});

describe('backslashes before a placeholder', () => {
  const registry = new ToolOutputReferenceRegistry();
  registry.set('current', 'tool0turn2', 'report.txt');
  const resolve = (input: string) => registry.resolve('current', input);

  it.each([
    ['\\{{tool0turn2}}', '{{tool0turn2}}'],
    ['\\\\{{tool0turn2}}', '\\report.txt'],
    ['\\\\\\{{tool0turn2}}', '\\{{tool0turn2}}'],
    ['\\\\\\\\{{tool0turn2}}', '\\\\report.txt'],
    ['type C:\\out\\\\{{tool0turn2}}', 'type C:\\out\\report.txt'],
    ['{{tool0turn2}}\\{{tool0turn2}}', 'report.txt{{tool0turn2}}'],
    ['{{tool0turn2}}\\\\{{tool0turn2}}', 'report.txt\\report.txt'],
    ['a\\b {{tool0turn2}} c\\\\d', 'a\\b report.txt c\\\\d'],
  ])('resolves %j to %j', (input, expected) => {
    expect(resolve(input)).toEqual({ resolved: expected, unresolved: [] });
  });

  it('looks up an unescaped key behind an even run and skips an escaped one', () => {
    expect(resolve('\\\\{{tool8turn8}} \\{{tool9turn9}}').unresolved).toEqual([
      'tool8turn8',
    ]);
  });

  it('resolves a reference after a long backslash run', () => {
    const run = '\\'.repeat(200_000);
    expect(resolve(`${run}{{tool0turn2}}`).resolved).toBe(
      `${'\\'.repeat(100_000)}report.txt`
    );
    expect(resolve(`{{tool0turn2}}${run}x`).resolved).toBe(`report.txt${run}x`);
  });
});

it('detaches only reference-bearing comparison inputs', () => {
  const plain = { command: 'initial', nested: { value: 'plain' } };
  const clone = jest.spyOn(globalThis, 'structuredClone');
  expect(snapshotResolvedArgs(plain)).toEqual({});
  expect(clone).not.toHaveBeenCalled();
  const input = {
    command: 'literal {{tool8turn8}}',
    nested: { value: '{{tool9turn9}}' },
  };
  const baseline = snapshotResolvedArgs(input);
  input.nested.value = '{{tool8turn8}}';
  expect(baseline).toEqual({
    command: 'literal {{tool8turn8}}',
    nested: { value: '{{tool9turn9}}' },
  });
  expect(clone).toHaveBeenCalledTimes(1);
});
it('preserves the reference-free fast path but detaches edited values', () => {
  const registry = new ToolOutputReferenceRegistry();
  const input = { command: 'initial', nested: [{ value: 'plain' }] };
  expect(registry.resolve('current', input).resolved).toBe(input);
  const edited = registry.resolve('current', input, {
    previouslyResolved: snapshotResolvedArgs(input),
  }).resolved;
  expect(edited).toEqual(input);
  expect(edited).not.toBe(input);
  expect(edited.nested[0]).not.toBe(input.nested[0]);
});
it('preserves serialized resolved fields and validates new serialized edits', () => {
  const registry = new ToolOutputReferenceRegistry();
  registry.set('current', 'tool0turn2', '{{tool8turn8}}');
  const first = registry.resolve(
    'current',
    JSON.stringify({ command: '{{tool0turn2}}', nested: { value: 'initial' } })
  ).resolved;
  const edited = registry.resolve(
    'current',
    JSON.stringify({
      command: '{{tool8turn8}}',
      nested: { value: '{{tool9turn9}}' },
    }),
    { previouslyResolved: first }
  );
  expect(JSON.parse(edited.resolved).command).toBe('{{tool8turn8}}');
  expect(edited.unresolved).toEqual(['tool9turn9']);
});

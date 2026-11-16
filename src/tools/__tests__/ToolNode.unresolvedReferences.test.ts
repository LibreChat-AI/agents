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
import { ToolOutputReferenceRegistry } from '../toolOutputReferences';
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
  approval = false
) {
  const mutations: string[] = [];
  const dispatched: string[] = [];
  const registry = new ToolOutputReferenceRegistry({ maxTotalSize: 100 });
  const makeTool = (name: string) =>
    tool(
      async ({ command }) => {
        mutations.push(command);
        return command;
      },
      {
        name,
        description: 'counter only, no external effects',
        schema: z.object({ command: z.string() }),
      }
    );
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      if (event !== 'on_tool_execute') return;
      const batch = data as t.ToolExecuteBatchRequest;
      batch.resolve(
        batch.toolCalls.map((request) => {
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
    toolOutputRegistry: registry,
    hookRegistry: hooks,
    humanInTheLoop: { enabled: approval },
    toolCallStepIds: new Map(
      ['bad', 'good', 'retry'].map((id) => [id, `step-${id}`])
    ),
  });
  return { node, registry, mutations, dispatched };
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
  expect(result.content).toContain('tool0turn2');
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

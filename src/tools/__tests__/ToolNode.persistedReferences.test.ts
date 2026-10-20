import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  annotateMessagesForLLM,
  nextReferenceTurn,
  ToolOutputReferenceRegistry,
} from '../toolOutputReferences';
import { createContentAggregator } from '@/stream';
import { formatAgentMessages } from '@/messages/format';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import * as events from '@/utils/events';
import { StandardGraph } from '@/graphs';
import { ToolNode } from '../ToolNode';
import { Run } from '@/run';

type Completion = { result: { id: string; tool_call: t.ProcessedToolCall } };

function createEchoTool(): StructuredToolInterface {
  return tool(async (input) => (input as { command: string }).command, {
    name: 'echo',
    description: 'Echo test tool',
    schema: z.object({ command: z.string() }),
  }) as unknown as StructuredToolInterface;
}

function captureCompletions(): Completion[] {
  const completions: Completion[] = [];
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data): Promise<void> => {
      if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
        completions.push(data as Completion);
      }
    });
  return completions;
}

function aiMsg(id: string, command: string): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ id, name: 'echo', args: { command } }],
  });
}

async function runBatch(
  node: ToolNode,
  id: string,
  command: string,
  runId = 'run-a'
): Promise<ToolMessage> {
  const out = (await node.invoke(
    { messages: [aiMsg(id, command)] },
    { configurable: { run_id: runId } }
  )) as { messages: ToolMessage[] };
  return out.messages[0];
}

describe('persisted tool output reference labels', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('puts the key and unresolved refs on the completed tool_call (direct path)', async () => {
    const completions = captureCompletions();
    const node = new ToolNode({
      tools: [createEchoTool()],
      toolCallStepIds: new Map([
        ['c1', 'step_1'],
        ['c2', 'step_2'],
      ]),
      toolOutputReferences: { enabled: true },
    });
    await runBatch(node, 'c1', 'first');
    await runBatch(node, 'c2', 'use {{tool0turn9}}');

    expect(completions[0].result.tool_call.ref).toBe('tool0turn0');
    expect(completions[0].result.tool_call.unresolved_refs).toBeUndefined();
    expect(completions[1].result.tool_call.ref).toBe('tool0turn1');
    expect(completions[1].result.tool_call.unresolved_refs).toEqual([
      'tool0turn9',
    ]);
  });

  it('omits ref fields when the feature is disabled', async () => {
    const completions = captureCompletions();
    const node = new ToolNode({
      tools: [createEchoTool()],
      toolCallStepIds: new Map([['c1', 'step_1']]),
    });
    await runBatch(node, 'c1', 'first');
    expect(completions[0].result.tool_call).not.toHaveProperty('ref');
    expect(completions[0].result.tool_call).not.toHaveProperty(
      'unresolved_refs'
    );
  });

  it('restores labels byte-for-byte after truncation and a persistence round trip', async () => {
    const completions = captureCompletions();
    const node = new ToolNode({
      tools: [createEchoTool()],
      maxToolResultChars: 200,
      toolCallStepIds: new Map([['c1', 'step_1']]),
      toolOutputReferences: { enabled: true },
    });
    const live = await runBatch(node, 'c1', 'x'.repeat(5000));
    expect((live.content as string).length).toBeLessThan(5000);

    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_1',
        stepIndex: 0,
        index: 0,
        type: 'tool_calls',
        stepDetails: {
          type: 'tool_calls',
          tool_calls: [{ id: 'c1', name: 'echo', args: '{}' }],
        },
        usage: null,
      } as unknown as t.RunStep,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: completions[0] as unknown as { result: t.ToolEndEvent },
    });
    const stored = JSON.parse(JSON.stringify(contentParts)) as t.MessageContentComplex[];

    const { messages } = formatAgentMessages(
      [
        { role: 'user', content: 'go' },
        { role: 'assistant', content: stored },
      ],
      undefined,
      undefined,
      undefined,
      { provider: Providers.OPENAI }
    );
    const restored = messages.find((m) => m._getType() === 'tool') as ToolMessage;
    expect(restored.additional_kwargs._refKey).toBe('tool0turn0');

    const registry = new ToolOutputReferenceRegistry();
    const wireLive = annotateMessagesForLLM([live], registry, 'run-b');
    const wireRestored = annotateMessagesForLLM([restored], registry, 'run-c');
    expect(wireRestored[0].content).toBe(wireLive[0].content);
    expect(wireRestored[0].content).toMatch(/^\[ref: tool0turn0\]\n/);
  });

  it('restores unresolved refs as additional_kwargs', () => {
    const { messages } = formatAgentMessages(
      [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'c1',
                name: 'echo',
                args: '{}',
                output: 'bad',
                ref: 'tool1turn3',
                unresolved_refs: ['tool0turn0'],
              },
            },
          ],
        },
      ],
      undefined,
      undefined,
      undefined,
      { provider: Providers.OPENAI }
    );
    const tm = messages.find((m) => m._getType() === 'tool') as ToolMessage;
    expect(tm.additional_kwargs).toMatchObject({
      _refKey: 'tool1turn3',
      _unresolvedRefs: ['tool0turn0'],
    });
    const [wire] = annotateMessagesForLLM([tm], new ToolOutputReferenceRegistry());
    expect(wire.content).toBe(
      '[ref: tool1turn3]\nbad\n[unresolved refs: tool0turn0]'
    );
    expect(
      (messages.find((m) => m._getType() === 'ai') as AIMessage).tool_calls?.[0]
    ).not.toHaveProperty('ref');
  });

  it('adds ref fields to early and eager event-driven completions', async () => {
    const completions = captureCompletions();
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(data as Completion);
          return;
        }
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          const batch = data as t.ToolExecuteBatchRequest;
          const results = batch.toolCalls.map((r) => ({
            toolCallId: r.id,
            status: 'success' as const,
            content: 'out',
          }));
          results.forEach((r) => batch.onResult?.(r));
          await new Promise((resolve) => setTimeout(resolve, 0));
          batch.resolve(results);
        }
      });
    const node = new ToolNode({
      tools: [createEchoTool()],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['c1', 'step_1']]),
      toolOutputReferences: { enabled: true },
    });
    const msg = await runBatch(node, 'c1', 'a');
    expect(msg.additional_kwargs._refKey).toBe('tool0turn0');
    expect(completions).toHaveLength(1);
    expect(completions[0].result.tool_call.ref).toBe('tool0turn0');

    completions.length = 0;
    const eager = new Map<string, t.EagerEventToolExecution>();
    eager.set('c2', {
      toolCallId: 'c2',
      toolName: 'echo',
      args: { command: 'b' },
      request: {
        id: 'c2',
        name: 'echo',
        args: { command: 'b' },
        stepId: 'step_2',
        turn: 1,
      },
      completionDispatched: true,
      promise: Promise.resolve({
        results: [{ toolCallId: 'c2', status: 'success', content: 'eager' }],
      }),
    });
    const eagerNode = new ToolNode({
      tools: [createEchoTool()],
      eventDrivenMode: true,
      eagerEventToolExecution: { enabled: true },
      eagerEventToolExecutions: eager,
      eagerEventToolUsageCount: new Map([['echo', 1]]),
      toolCallStepIds: new Map([['c2', 'step_2']]),
      toolOutputReferences: { enabled: true },
    });
    await runBatch(eagerNode, 'c2', 'b');
    expect(completions).toHaveLength(1);
    expect(completions[0].result.tool_call).toMatchObject({
      output: 'eager',
      ref: 'tool0turn0',
    });
  });
});

describe('turn counter continuity across runs', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  const toolMsg = (key?: string): ToolMessage =>
    new ToolMessage({
      content: 'x',
      tool_call_id: 'id',
      additional_kwargs: key == null ? {} : { _refKey: key },
    });

  it('nextReferenceTurn is one past the highest restored turn', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      toolMsg('tool0turn0'),
      toolMsg('tool3turn4'),
      toolMsg('tool1turn2'),
      toolMsg(),
      toolMsg('garbage'),
    ];
    expect(nextReferenceTurn(history)).toBe(5);
    expect(nextReferenceTurn([new HumanMessage('hi')])).toBe(0);
  });

  it('seedTurn raises but never lowers the counter', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.seedTurn('r', 3);
    expect(registry.nextTurn('r')).toBe(3);
    registry.seedTurn('r', 1);
    expect(registry.nextTurn('r')).toBe(4);
  });

  it('a placeholder naming a previous run key is unresolved, not substituted', async () => {
    captureCompletions();
    const node = new ToolNode({
      tools: [createEchoTool()],
      toolCallStepIds: new Map([['c1', 'step_1']]),
      toolOutputReferences: { enabled: true },
    });
    const previous = toolMsg('tool0turn0');
    node
      ._unsafeGetToolOutputRegistry()!
      .seedTurn('run-b', nextReferenceTurn([previous]));
    const msg = await runBatch(node, 'c1', 'echo {{tool0turn0}}', 'run-b');
    expect(msg.additional_kwargs._refKey).toBe('tool0turn1');
    expect(msg.additional_kwargs._unresolvedRefs).toEqual(['tool0turn0']);
    expect(msg.content).toBe('echo {{tool0turn0}}');
    const [wire] = annotateMessagesForLLM(
      [msg],
      node._unsafeGetToolOutputRegistry()
    );
    expect(wire.content).toContain('[unresolved refs: tool0turn0]');
  });

  it('Run seeds the registry from the input history', async () => {
    const run = await Run.create<t.IState>({
      runId: 'seed-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'x',
            maxContextTokens: 8000,
          },
        ],
      },
      toolOutputReferences: { enabled: true },
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel(['ok'], 10);
    await run.processStream(
      {
        messages: [
          new HumanMessage('q'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 't1', name: 'echo', args: {} }],
          }),
          toolMsg('tool2turn6'),
          new HumanMessage('next'),
        ],
      },
      {
        configurable: { thread_id: 'seed-thread' },
        version: 'v2',
      }
    );
    const registry = (run.Graph as StandardGraph).getOrCreateToolOutputRegistry();
    expect(registry?.nextTurn('seed-run')).toBe(7);
  });
});

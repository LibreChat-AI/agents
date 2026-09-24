import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { GraphInterrupt } from '@langchain/langgraph';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import {
  ToolOutputReferenceRegistry,
  annotateMessagesForLLM,
} from '../toolOutputReferences';
import { restoreToolReplayConfig } from '../toolBatchReplay';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';
import { GraphEvents } from '@/common';

afterEach(() => jest.restoreAllMocks());

async function capturePause(
  node: ToolNode,
  input: object,
  config: RunnableConfig
): Promise<unknown> {
  try {
    await node.invoke(input, config);
  } catch (error) {
    if (error instanceof GraphInterrupt) {
      return JSON.parse(JSON.stringify(error.interrupts[0].value));
    }
    throw error;
  }
  throw new Error('Expected an interrupt');
}

function replayConfig(payload: unknown): RunnableConfig {
  const configurable = { thread_id: 'checkpoint-replay' };
  restoreToolReplayConfig(configurable, 'pause', payload);
  return { configurable };
}

describe('ToolNode checkpoint authority', () => {
  it.each([false, true])('keeps concurrently checkpointed references while replaying its frozen inputs (claimed=%s)', async (claimed) => {
    let release: () => void = () => { throw new Error('not initialized'); };
    const barrier = new Promise<void>((resolve) => { release = resolve; });
    let shouldPause = true;
    const work = tool(async ({ value }) => {
      if (shouldPause) {
        await barrier;
        throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
      }
      return value;
    }, { name: 'work', description: 'work', schema: z.object({ value: z.string() }) });
    const sibling = tool(async () => 'sibling-result', { name: 'sibling', description: 'sibling', schema: z.object({}) });
    const registry = new ToolOutputReferenceRegistry();
    const config = { configurable: { thread_id: 'checkpoint-replay', run_id: 'shared' } };
    const input = { messages: [new AIMessage({ id: 'batch', content: '', tool_calls: [
      { id: 'work', name: 'work', args: { value: '{{tool0turn1}}' } },
    ] })] };
    const paused = capturePause(new ToolNode({ tools: [work], toolOutputRegistry: registry }), input, config);
    await new ToolNode({ tools: [sibling], toolOutputRegistry: registry }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [
      { id: 'sibling', name: 'sibling', args: {} },
    ] })] }, config);
    release();
    const checkpoint = await paused;
    const sharedState = registry.snapshotState('shared');
    expect(sharedState.turnCounter).toBe(2);
    const restoredRegistry = new ToolOutputReferenceRegistry();
    restoredRegistry.restoreState('shared', sharedState);
    const resume = replayConfig(checkpoint);
    resume.configurable = { ...resume.configurable, run_id: 'shared' };
    shouldPause = false;
    const result = await new ToolNode({ tools: [work], toolOutputRegistry: restoredRegistry, onToolCallsClaimed: claimed ? async () => { await Promise.resolve(); } : undefined }).invoke(input, resume) as { messages: ToolMessage[] };
    expect(result.messages[0].content).toBe('{{tool0turn1}}');
    expect(restoredRegistry.get('shared', 'tool0turn1')).toBe('sibling-result');
    expect(restoredRegistry.nextTurn('shared')).toBe(2);
  });

  it.each([undefined, ''])('restores an ID-less completed sibling (%j) without repeating it', async (id) => {
    let executions = 0;
    let shouldPause = true;
    const work = tool(async () => { executions += 1; return `result-${executions}`; }, { name: 'work', description: 'work', schema: z.object({}) });
    const pause = tool(async () => {
      if (shouldPause) throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
      return 'resumed';
    }, { name: 'pause', description: 'pause', schema: z.object({}) });
    const alreadyDone = tool(async () => 'done', { name: 'done', description: 'done', schema: z.object({}) });
    const input = { messages: [new AIMessage({ id: 'batch', content: '', tool_calls: [
      { id: 'done', name: 'done', args: {} }, { id, name: 'work', args: {} }, { id, name: 'work', args: {} }, { id: 'pause', name: 'pause', args: {} },
    ] })] };
    const checkpoint = await capturePause(new ToolNode({ tools: [work, pause, alreadyDone] }), input, { configurable: { thread_id: 'checkpoint-replay' } });
    expect(executions).toBe(2);
    shouldPause = false;
    const result = await new ToolNode({ tools: [work, pause, alreadyDone] }).invoke({ messages: [
      ...input.messages, new ToolMessage({ tool_call_id: 'done', content: 'done' }),
    ] }, replayConfig(checkpoint));
    expect(executions).toBe(2);
    expect(JSON.stringify(result)).toContain('result-1');
    expect(JSON.stringify(result)).toContain('result-2');
  });

  it.each([false, true])(
    'restores an older checkpoint despite newer cached output (append=%s)',
    async (append) => {
      let generation = 'original';
      let executions = 0;
      let shouldPause = true;
      const work = tool(
        async () => {
          executions += 1;
          return generation;
        },
        { name: 'work', description: 'work', schema: z.object({}) }
      );
      const pause = tool(
        async () => {
          if (shouldPause) {
            throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
          }
          return 'resumed';
        },
        { name: 'pause', description: 'pause', schema: z.object({}) }
      );
      const node = new ToolNode({ tools: [work, pause] });
      const input = {
        messages: [
          new AIMessage({
            id: 'batch',
            content: '',
            tool_calls: [
              { id: 'work', name: 'work', args: {} },
              { id: 'pause', name: 'pause', args: {} },
            ],
          }),
        ],
      };
      const older = await capturePause(node, input, {
        configurable: { thread_id: 'checkpoint-replay' },
      });
      const emptyCheckpoint = replayConfig(older);
      const state = emptyCheckpoint.configurable
        ?.__librechat_tool_batch_replay as { records: object[] };
      state.records = [];
      generation = 'newer';
      await capturePause(node, input, emptyCheckpoint);
      expect(executions).toBe(2);
      shouldPause = false;
      const result = await node.invoke(
        append
          ? {
            messages: [
              ...input.messages,
              new HumanMessage('More instructions'),
            ],
          }
          : input,
        replayConfig(older)
      );
      expect(JSON.stringify(result)).toContain('original');
      expect(JSON.stringify(result)).not.toContain('newer');
      expect(executions).toBe(2);
    }
  );

  it.each(['same-run', 'different-run', undefined])(
    'restores reference content and counters under resumed scope %s',
    async (resumedRunId) => {
      let shouldPause = true;
      const work = tool(
        async ({ value, pause }) => {
          if (pause && shouldPause) {
            throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
          }
          return value;
        },
        {
          name: 'work',
          description: 'work',
          schema: z.object({ value: z.string(), pause: z.boolean() }),
        }
      );
      const makeNode = (registry: ToolOutputReferenceRegistry) =>
        new ToolNode({ tools: [work], toolOutputRegistry: registry });
      const original = makeNode(new ToolOutputReferenceRegistry());
      const config = {
        configurable: { thread_id: 'checkpoint-replay', run_id: 'same-run' },
      };
      const earlier = (await original.invoke(
        {
          messages: [
            new AIMessage({
              id: 'earlier',
              content: '',
              tool_calls: [
                {
                  id: 'earlier',
                  name: 'work',
                  args: { value: 'old-value', pause: false },
                },
              ],
            }),
          ],
        },
        config
      )) as { messages: ToolMessage[] };
      const input = {
        messages: [
          new AIMessage({
            id: 'batch',
            content: '',
            tool_calls: [
              {
                id: 'completed',
                name: 'work',
                args: { value: 'cached-value', pause: false },
              },
              {
                id: 'pending',
                name: 'work',
                args: { value: '{{tool0turn1}}', pause: true },
              },
            ],
          }),
        ],
      };
      const checkpoint = await capturePause(original, input, config);
      const registry = new ToolOutputReferenceRegistry();
      const fresh = makeNode(registry);
      shouldPause = false;
      const resume = replayConfig(checkpoint);
      resume.configurable = { ...resume.configurable, run_id: resumedRunId };
      const result = (await fresh.invoke(input, resume)) as {
        messages: ToolMessage[];
      };
      const completed = result.messages.find(
        (message) => message.tool_call_id === 'completed'
      )!;
      const pending = result.messages.find(
        (message) => message.tool_call_id === 'pending'
      )!;
      const scope = pending.additional_kwargs._refScope as string;
      expect(registry.get(scope, 'tool0turn0')).toBe('old-value');
      expect(pending.additional_kwargs._refKey).not.toBe('tool0turn0');
      expect(pending.additional_kwargs._refKey).toBe('tool1turn1');
      expect(pending.content).toBe('{{tool0turn1}}');
      expect(completed.additional_kwargs._refScope).toBe(scope);
      expect(
        annotateMessagesForLLM([completed], registry, resumedRunId)[0].content
      ).toContain('[ref: tool0turn1]');
      expect(registry.resolve(scope, '{{tool0turn1}}').resolved).toBe(
        'cached-value'
      );
      if (resumedRunId === 'same-run') {
        expect(
          annotateMessagesForLLM(earlier.messages, registry, resumedRunId)[0]
            .content
        ).toContain('old-value');
        expect(registry.resolve(scope, '{{tool0turn0}}').resolved).toBe(
          'old-value'
        );
      }
    }
  );

  it.each([false, true])(
    'restores completed and pending turns in a fresh runtime (hooks=%s)',
    async (hooks) => {
      const observed: Array<{ id: string; turn: number }> = [];
      const completions: Array<{ index: number; tool_call: { id: string } }> =
        [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
            completions.push(
              (data as { result: (typeof completions)[number] }).result
            );
          }
          return true;
        });
      let shouldPause = true;
      const work = tool(
        async ({ pause }, config) => {
          const call = config.toolCall as { id: string; turn: number };
          observed.push({ id: call.id, turn: call.turn });
          if (pause && shouldPause) {
            throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
          }
          return `turn-${call.turn}`;
        },
        {
          name: 'work',
          description: 'work',
          schema: z.object({ pause: z.boolean() }),
        }
      );
      const makeNode = () => {
        const registry = new HookRegistry();
        registry.register('PreToolUse', {
          hooks: [async () => ({ decision: 'allow' })],
        });
        return new ToolNode({
          tools: [work],
          hookRegistry: hooks ? registry : undefined,
          toolCallStepIds: new Map(
            ['earlier', 'completed', 'pending'].map((id) => [id, `step-${id}`])
          ),
        });
      };
      const node = makeNode();
      const config = { configurable: { thread_id: 'checkpoint-replay' } };
      await node.invoke(
        {
          messages: [
            new AIMessage({
              id: 'earlier',
              content: '',
              tool_calls: [
                { id: 'earlier', name: 'work', args: { pause: false } },
              ],
            }),
          ],
        },
        config
      );
      const input = {
        messages: [
          new AIMessage({
            id: 'batch',
            content: '',
            tool_calls: [
              { id: 'completed', name: 'work', args: { pause: false } },
              { id: 'pending', name: 'work', args: { pause: true } },
            ],
          }),
        ],
      };
      const checkpoint = await capturePause(node, input, config);
      expect(observed).toEqual([
        { id: 'earlier', turn: 0 },
        { id: 'completed', turn: 1 },
        { id: 'pending', turn: 2 },
      ]);
      shouldPause = false;
      const fresh = makeNode();
      const resumed = await fresh.invoke(input, replayConfig(checkpoint));
      expect(observed).toEqual([
        { id: 'earlier', turn: 0 },
        { id: 'completed', turn: 1 },
        { id: 'pending', turn: 2 },
        { id: 'pending', turn: 2 },
      ]);
      expect(JSON.stringify(resumed)).toContain('turn-1');
      expect(JSON.stringify(resumed)).toContain('turn-2');
      expect(fresh.getToolUsageCounts().get('work')).toBe(3);
      expect(
        completions.map(({ index, tool_call }) => ({ id: tool_call.id, index }))
      ).toEqual([
        { id: 'earlier', index: 0 },
        { id: 'completed', index: 1 },
        { id: 'pending', index: 2 },
      ]);
    }
  );
});

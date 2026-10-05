import { expect, test, describe } from '@jest/globals';
import { _reindexToolCallStream } from './index';

/**
 * Coverage for OpenAI-compatible providers that stream parallel tool calls with
 * unreliable `index` (e.g. Ollama, ollama/ollama#15457), which makes LangChain
 * merge siblings and drop their args.
 */

const TOOL = 'echo_mcp_everything-test';

type ToolCallDelta = {
  index?: number;
  id?: string;
  type?: 'function';
  function?: { name?: string; arguments?: string };
};

function chunkWith(toolCalls: ToolCallDelta[]): Record<string, unknown> {
  return {
    id: 'chatcmpl-1',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'qwen',
    choices: [{ index: 0, delta: { tool_calls: toolCalls }, finish_reason: null }],
  };
}

async function reindex(chunks: ToolCallDelta[][]): Promise<ToolCallDelta[]> {
  async function* source(): AsyncGenerator<Record<string, unknown>> {
    for (const toolCalls of chunks) {
      yield chunkWith(toolCalls);
    }
  }
  const flat: ToolCallDelta[] = [];
  for await (const chunk of _reindexToolCallStream(
    source() as unknown as AsyncIterable<never>,
  )) {
    const choices = (chunk as unknown as Record<string, unknown>)
      .choices as Array<{ delta?: { tool_calls?: ToolCallDelta[] } }>;
    for (const choice of choices) {
      flat.push(...(choice.delta?.tool_calls ?? []));
    }
  }
  return flat;
}

/**
 * Merge deltas the way an `index`-keyed accumulator (LangChain) does:
 * concatenate args per index and parse. Returns parsed args per call by index,
 * or `null` when the accumulated args aren't valid JSON.
 */
function mergeByIndex(deltas: ToolCallDelta[]): Array<unknown> {
  const byIndex = new Map<number, { args: string }>();
  for (const d of deltas) {
    const index = typeof d.index === 'number' ? d.index : 0;
    const bucket = byIndex.get(index) ?? { args: '' };
    bucket.args += d.function?.arguments ?? '';
    byIndex.set(index, bucket);
  }
  return [...byIndex.keys()]
    .sort((a, b) => a - b)
    .map((index) => {
      try {
        return JSON.parse(byIndex.get(index)!.args) as unknown;
      } catch {
        return null;
      }
    });
}

describe('streamed tool-call index repair', () => {
  test('repairs index: 0 on every call, fragmented args (ollama/ollama#15457)', async () => {
    const out = await reindex([
      [{ index: 0, id: 'a', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ index: 0, function: { arguments: '{"message":' } }],
      [{ index: 0, function: { arguments: '"alpha"}' } }],
      [{ index: 0, id: 'b', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ index: 0, function: { arguments: '{"message":' } }],
      [{ index: 0, function: { arguments: '"beta"}' } }],
      [{ index: 0, id: 'c', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ index: 0, function: { arguments: '{"message":' } }],
      [{ index: 0, function: { arguments: '"gamma"}' } }],
    ]);
    expect(mergeByIndex(out)).toEqual([
      { message: 'alpha' },
      { message: 'beta' },
      { message: 'gamma' },
    ]);
  });

  test('repairs missing index, fragmented args (ollama/ollama#7881)', async () => {
    const out = await reindex([
      [{ id: 'a', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ function: { arguments: '{"message":"alpha"}' } }],
      [{ id: 'b', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ function: { arguments: '{"message":"beta"}' } }],
    ]);
    expect(mergeByIndex(out)).toEqual([{ message: 'alpha' }, { message: 'beta' }]);
  });

  test('leaves well-behaved streams (distinct indices) untouched', async () => {
    const input: ToolCallDelta[][] = [
      [{ index: 0, id: 'a', type: 'function', function: { name: TOOL, arguments: '{"message":' } }],
      [{ index: 0, function: { arguments: '"alpha"}' } }],
      [{ index: 1, id: 'b', type: 'function', function: { name: TOOL, arguments: '{"message":"beta"}' } }],
      [{ index: 2, id: 'c', type: 'function', function: { name: TOOL, arguments: '{"message":"gamma"}' } }],
    ];
    const out = await reindex(input.map((c) => JSON.parse(JSON.stringify(c))));
    expect(out.map((tc) => tc.index)).toEqual([0, 0, 1, 2]);
    expect(mergeByIndex(out)).toEqual([
      { message: 'alpha' },
      { message: 'beta' },
      { message: 'gamma' },
    ]);
  });

  test('without repair, fragmented args + shared index 0 collapse (documents the bug)', () => {
    const raw: ToolCallDelta[] = [
      { index: 0, id: 'a', type: 'function', function: { name: TOOL, arguments: '' } },
      { index: 0, function: { arguments: '{"message":"alpha"}' } },
      { index: 0, id: 'b', type: 'function', function: { name: TOOL, arguments: '' } },
      { index: 0, function: { arguments: '{"message":"beta"}' } },
    ];
    // Index-keyed merge with no repair collapses both calls into index 0, whose
    // concatenated args are not valid JSON.
    expect(mergeByIndex(raw)).toEqual([null]);
  });

  test('repairs id-less starts that reuse an index, via the name signal', async () => {
    // No tool-call ids at all; each call start carries function.name and the
    // provider stamps index 0 on every delta.
    const out = await reindex([
      [{ index: 0, type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ index: 0, function: { arguments: '{"message":"alpha"}' } }],
      [{ index: 0, type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ index: 0, function: { arguments: '{"message":"beta"}' } }],
    ]);
    expect(mergeByIndex(out)).toEqual([{ message: 'alpha' }, { message: 'beta' }]);
  });

  test('repairs index-less argument fragments when starts carry distinct indices', async () => {
    const out = await reindex([
      [{ index: 0, id: 'a', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ function: { arguments: '{"message":"alpha"}' } }],
      [{ index: 1, id: 'b', type: 'function', function: { name: TOOL, arguments: '' } }],
      [{ function: { arguments: '{"message":"beta"}' } }],
    ]);
    expect(mergeByIndex(out)).toEqual([{ message: 'alpha' }, { message: 'beta' }]);
  });

  test('keeps repair state separate per choice (n > 1)', async () => {
    // Two choices each open a call at provider index 0. Shared state would
    // mis-detect a reused index and bump choice 1 to index 1.
    async function* source(): AsyncGenerator<Record<string, unknown>> {
      yield {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'qwen',
        choices: [
          { index: 0, delta: { tool_calls: [{ index: 0, id: 'x', type: 'function', function: { name: TOOL, arguments: '{"message":"alpha"}' } }] }, finish_reason: null },
          { index: 1, delta: { tool_calls: [{ index: 0, id: 'y', type: 'function', function: { name: TOOL, arguments: '{"message":"beta"}' } }] }, finish_reason: null },
        ],
      };
    }
    const choiceIndices = new Map<number, number[]>();
    for await (const chunk of _reindexToolCallStream(
      source() as unknown as AsyncIterable<never>,
    )) {
      const choices = (chunk as unknown as Record<string, unknown>)
        .choices as Array<{ index: number; delta?: { tool_calls?: ToolCallDelta[] } }>;
      for (const choice of choices) {
        const indices = choiceIndices.get(choice.index) ?? [];
        for (const tc of choice.delta?.tool_calls ?? []) {
          indices.push(tc.index as number);
        }
        choiceIndices.set(choice.index, indices);
      }
    }
    expect(choiceIndices.get(0)).toEqual([0]);
    expect(choiceIndices.get(1)).toEqual([0]);
  });
});

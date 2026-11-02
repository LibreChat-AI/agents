import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ToolMessage } from '@langchain/core/messages';
import { ToolOutputReferenceRegistry } from '../toolOutputReferences';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

const readerName = 'read_tool_output';
const readerConfig = { enabled: true, readTool: true };

interface Excerpt {
  content?: string;
  error?: string;
  start: number;
  end: number;
  retainedChars: number;
  nextOffset: number | null;
  sourceCompleteness: string;
}

async function invoke(
  node: ToolNode,
  calls: ToolCall[],
  runId = 'reader-run'
): Promise<ToolMessage[]> {
  const result = (await node.invoke(
    { messages: [new AIMessage({ content: '', tool_calls: calls })] },
    { configurable: { run_id: runId } }
  )) as { messages: ToolMessage[] };
  return result.messages;
}

function readCall(
  reference = 'tool0turn0',
  start = 0,
  maxChars = 4000,
  id = 'read'
): ToolCall {
  return { id, name: readerName, args: { reference, start, maxChars } };
}

describe('optional retained tool output reader', () => {
  it('retrieves an omitted command excerpt without invoking the producer again', async () => {
    let producerCalls = 0;
    const sentinel = 'MIDDLE_SENTINEL';
    const source = 'x'.repeat(20000) + sentinel + 'y'.repeat(20000);
    const producer = tool(async () => {
      producerCalls++;
      return source;
    }, { name: 'command', description: 'Synthetic command result', schema: z.object({}) });
    const node = new ToolNode({
      tools: [producer], maxToolResultChars: 512, toolOutputReferences: readerConfig,
    });
    const [first] = await invoke(node, [{ id: 'produce', name: 'command', args: {} }]);
    expect(first.content).not.toContain(sentinel);
    const [result] = await invoke(node, [readCall('tool0turn0', 20000, sentinel.length)]);
    expect(JSON.parse(String(result.content))).toMatchObject({
      content: sentinel, start: 20000, end: 20000 + sentinel.length,
      retainedChars: source.length, sourceCompleteness: 'unknown',
    });
    expect(String(result.content).length).toBeLessThanOrEqual(512);
    expect(result.additional_kwargs._refKey).toBeUndefined();
    expect(node._unsafeGetToolOutputRegistry()?.get('reader-run', 'tool0turn1')).toBeUndefined();
    expect(producerCalls).toBe(1);
  });

  it.each([undefined, false])('does not install a reader with readTool=%s', async (readTool) => {
    const config = { enabled: true, readTool };
    const node = new ToolNode({ tools: [], toolOutputReferences: config });
    const [result] = await invoke(node, [readCall()]);
    expect(result.status).toBe('error');
  });

  it('uses post-hook output and never registers a rewritten reader result', async () => {
    const hooks = new HookRegistry();
    hooks.register('PostToolUse', {
      hooks: [async (input) => {
        if (input.toolName === readerName) {
          expect(JSON.parse(String(input.toolOutput))).toMatchObject({ content: 'approved replacement' });
          expect(String(input.toolOutput)).not.toContain('raw private text');
        }
        return { updatedOutput: input.toolName === 'producer' ? 'approved replacement' : 'reader hook result' };
      }],
    });
    const producer = tool(async () => 'raw private text', {
      name: 'producer', description: 'Producer', schema: z.object({}),
    });
    const node = new ToolNode({ tools: [producer], hookRegistry: hooks, toolOutputReferences: readerConfig });
    await invoke(node, [{ id: 'produce', name: 'producer', args: {} }]);
    expect(node._unsafeGetToolOutputRegistry()?.get('reader-run', 'tool0turn0')).toBe('approved replacement');
    const [result] = await invoke(node, [readCall()]);
    expect(result.content).toBe('reader hook result');
    expect(result.additional_kwargs._refKey).toBeUndefined();
    expect(node._unsafeGetToolOutputRegistry()?.get('reader-run', 'tool0turn1')).toBeUndefined();
  });

  it('does not expand reference syntax before hooks or execution', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('reader-run', 'tool0turn0', 'never in arguments');
    const observed: object[] = [];
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', { hooks: [async (input) => {
      observed.push(input.toolInput);
      return { decision: 'allow' };
    }] });
    const node = new ToolNode({ tools: [], hookRegistry: hooks, toolOutputRegistry: registry, toolOutputReferences: readerConfig });
    const [result] = await invoke(node, [readCall('{{tool0turn0}}')]);
    expect(observed).toEqual([{ reference: '{{tool0turn0}}', start: 0, maxChars: 4000 }]);
    expect(String(result.content)).not.toContain('never in arguments');
    expect(JSON.parse(String(result.content))).toEqual({ error: 'invalid_reference' });
  });

  it('uses the pre-batch snapshot after a sibling has registered its output', async () => {
    let release = (): void => { throw new Error('Gate not initialized'); };
    const completed = new Promise<void>((resolve) => { release = resolve; });
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', { hooks: [async (input) => {
      if (input.toolName === readerName) await completed;
      return { decision: 'allow' };
    }] });
    hooks.register('PostToolUse', { hooks: [async (input) => {
      if (input.toolName === 'producer') release();
      return {};
    }] });
    const producer = tool(async () => 'sibling result', {
      name: 'producer', description: 'Producer', schema: z.object({}),
    });
    const node = new ToolNode({ tools: [producer], hookRegistry: hooks, toolOutputReferences: readerConfig });
    const results = await invoke(node, [readCall('tool1turn0'), { id: 'produce', name: 'producer', args: {} }]);
    expect(JSON.parse(String(results[0].content))).toEqual({ error: 'unavailable' });
    const [next] = await invoke(node, [readCall('tool1turn0', 0, 4000, 'next')]);
    expect(JSON.parse(String(next.content))).toMatchObject({ content: 'sibling result' });
  });

  it('isolates runs and reports evicted or unknown content as unavailable', async () => {
    const registry = new ToolOutputReferenceRegistry({ maxOutputSize: 32, maxTotalSize: 32 });
    registry.set('owner', 'tool0turn0', 'first'.repeat(6));
    registry.set('owner', 'tool0turn1', 'second'.repeat(5));
    const node = new ToolNode({ tools: [], toolOutputRegistry: registry, toolOutputReferences: readerConfig });
    for (const [runId, reference] of [['foreign', 'tool0turn1'], ['owner', 'tool0turn0'], ['owner', 'tool9turn9']]) {
      const [result] = await invoke(node, [readCall(reference)], runId);
      expect(JSON.parse(String(result.content))).toEqual({ error: 'unavailable' });
    }
    const [result] = await invoke(node, [readCall('tool0turn1')], 'owner');
    expect(JSON.parse(String(result.content))).toMatchObject({ content: 'second'.repeat(5) });
    expect(registry.snapshotState('owner').entries).toEqual([{ key: 'tool0turn1', value: 'second'.repeat(5) }]);
  });

  it.each(['', 'ascii '.repeat(100), '\r\n"\\\u0000'.repeat(30), '😀漢字'.repeat(60), '\ud800x\udc00'.repeat(30), '{{tool9turn9}}\nEOF\n\'$(touch forbidden)'.repeat(15)])(
    'returns exact retained code units through bounded JSON pages (%#)', async (source) => {
      const registry = new ToolOutputReferenceRegistry();
      registry.set('reader-run', 'tool0turn0', source);
      const node = new ToolNode({ tools: [], toolOutputRegistry: registry, toolOutputReferences: readerConfig, maxToolResultChars: 256 });
      let start = 0;
      let reconstructed = '';
      for (let page = 0; page < 100; page++) {
        const [result] = await invoke(node, [readCall('tool0turn0', start, 1000, `page-${page}`)]);
        expect(String(result.content).length).toBeLessThanOrEqual(256);
        const excerpt = JSON.parse(String(result.content)) as Excerpt;
        expect(excerpt.error).toBeUndefined();
        expect(excerpt.start).toBe(start);
        expect(excerpt.content).toBe(source.slice(start, excerpt.end));
        reconstructed += excerpt.content;
        if (excerpt.nextOffset === null) break;
        expect(excerpt.nextOffset).toBeGreaterThan(start);
        start = excerpt.nextOffset;
      }
      expect(reconstructed).toBe(source);
      expect(registry.snapshotState('reader-run').entries).toHaveLength(1);
    }
  );

  it('finishes an exact-fit final page when null nextOffset is smaller than its numeric value', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('reader-run', 'tool0turn0', 'x'.repeat(200000));
    const expected = { reference: 'tool0turn0', start: 199880, end: 200000, retainedChars: 200000,
      nextOffset: null, unit: 'utf16', sourceCompleteness: 'unknown', content: 'x'.repeat(120) };
    const node = new ToolNode({ tools: [], toolOutputRegistry: registry, toolOutputReferences: readerConfig,
      maxToolResultChars: JSON.stringify(expected).length });
    const [result] = await invoke(node, [readCall('tool0turn0', 199880, 120)]);
    expect(JSON.parse(String(result.content))).toEqual(expected);
  });

  it.each([1, 5])('rejects a split-pair or out-of-range start of %s', async (start) => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('reader-run', 'tool0turn0', '😀ok');
    const node = new ToolNode({ tools: [], toolOutputRegistry: registry, toolOutputReferences: readerConfig });
    const [result] = await invoke(node, [readCall('tool0turn0', start)]);
    expect(JSON.parse(String(result.content))).toEqual({ error: 'invalid_range' });
  });

  it.each([0, 255, 256.5, Infinity, NaN])('rejects an unsupported reader result budget of %s', (maxToolResultChars) => {
    expect(() => new ToolNode({ tools: [], toolOutputReferences: readerConfig, maxToolResultChars })).toThrow('at least 256');
  });

  it('reports no progress when a requested character count cannot fit one complete pair', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('reader-run', 'tool0turn0', '😀');
    const node = new ToolNode({ tools: [], toolOutputRegistry: registry, toolOutputReferences: readerConfig });
    const [result] = await invoke(node, [readCall('tool0turn0', 0, 1)]);
    expect(JSON.parse(String(result.content))).toEqual({ error: 'excerpt_budget_too_small' });
  });

  it('reads restored retained state without claiming unavailable clipped text', async () => {
    const original = new ToolOutputReferenceRegistry({ maxOutputSize: 12 });
    original.set('reader-run', 'tool0turn0', 'retainedtext-lost-suffix');
    const restored = new ToolOutputReferenceRegistry({ maxOutputSize: 12 });
    restored.restoreState('reader-run', original.snapshotState('reader-run'));
    const node = new ToolNode({ tools: [], toolOutputRegistry: restored, toolOutputReferences: readerConfig });
    const [result] = await invoke(node, [readCall()]);
    expect(JSON.parse(String(result.content))).toMatchObject({ content: 'retainedtext', retainedChars: 12, sourceCompleteness: 'unknown' });
    expect(String(result.content)).not.toContain('lost-suffix');
  });

  it('honors denied pre-hooks without returning retained text or making a reference', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('reader-run', 'tool0turn0', 'DENIED_PRIVATE_TEXT');
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'deny', reason: 'Reader denied' })] });
    const node = new ToolNode({ tools: [], toolOutputRegistry: registry, toolOutputReferences: readerConfig, hookRegistry: hooks });
    const [result] = await invoke(node, [readCall()]);
    expect(String(result.content)).not.toContain('DENIED_PRIVATE_TEXT');
    expect(result.status).toBe('error');
    expect(registry.snapshotState('reader-run').entries).toHaveLength(1);
  });
});

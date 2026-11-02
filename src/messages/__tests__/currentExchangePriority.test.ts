import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ContextPruningConfig } from '@/types/graph';
import type { TokenCounter } from '@/types/run';
import { setProviderMessageProvenance } from '@/messages/provenance';
import { createPruneMessages } from '@/messages/prune';

const counter: TokenCounter = (message) => Math.ceil((
  JSON.stringify(message.content).length
  + (message.getType() === 'ai' ? JSON.stringify((message as AIMessage).tool_calls).length : 0)
) / 4) + 4;
const text = (label: string, size = 5000): string => `${label}:${'x'.repeat(size / 2)}:MIDDLE:${'y'.repeat(size / 2)}`;
const call = (id: string, args = { path: id }): AIMessage => new AIMessage({
  content: '', tool_calls: [{ id, name: 'read_file', args, type: 'tool_call' }],
});
const output = (id: string, content = text(id)): ToolMessage => new ToolMessage({
  content, tool_call_id: id, name: 'read_file',
});
const history = (): BaseMessage[] => [
  new HumanMessage('TASK: preserve requirements'), call('old'), output('old'), call('fresh'), output('fresh'),
];

function setup(config?: ContextPruningConfig, overrides: {
  maxTokens?: number; instructionTokens?: number; calibrationRatio?: number; summarize?: boolean;
} = {}) {
  const indexTokenCountMap: Record<string, number | undefined> = {};
  const prune = createPruneMessages({
    maxTokens: overrides.maxTokens ?? 16000, startIndex: 0, tokenCounter: counter,
    indexTokenCountMap, contextPruningConfig: config,
    maxToolResultChars: 12000, fadingTier: { v: 2, budgetTokens: 170, masked: true, latched: true },
    getInstructionTokens: () => overrides.instructionTokens ?? 500,
    calibrationRatio: overrides.calibrationRatio ?? 1, summarizationEnabled: overrides.summarize,
  });
  return { prune, indexTokenCountMap };
}

function project(canonical: BaseMessage[], config?: ContextPruningConfig, overrides?: Parameters<typeof setup>[1]) {
  const { prune } = setup(config, overrides);
  const messages = [...canonical];
  const result = prune({ messages, canonicalMessages: canonical });
  return { ...result, messages };
}

const enabled = { enabled: true, currentExchangePriority: true };

describe('fitting current tool exchange priority', () => {
  it('restores a fitting latest exchange while old results remain faded and canonical history is unchanged', () => {
    const canonical = history();
    const before = JSON.stringify(canonical);
    const baseline = project(canonical, { enabled: true });
    const result = project(canonical, enabled);
    expect((baseline.messages[4].content as string).length).toBeLessThan(300);
    expect(result.messages[4].content).toBe(canonical[4].content);
    expect(result.messages[4].content).toContain('MIDDLE');
    expect(result.messages[2].content).toBe(baseline.messages[2].content);
    expect(result.context).toContain(result.messages[3]);
    expect(result.context).toContain(result.messages[4]);
    expect(result.context).toContain(canonical[0]);
    expect(JSON.stringify(canonical)).toBe(before);
    expect(result.prePruneContextTokens! + result.effectiveInstructionTokens! + 3).toBeLessThanOrEqual(result.contextBudget!);
    expect(result.indexTokenCountMap[4]).toBe(counter(result.messages[4]));
  });

  it('keeps omitted and false settings identical and requires enabled pruning', () => {
    const canonical = history();
    const absent = project(canonical, { enabled: true });
    const disabled = project(canonical, { enabled: true, currentExchangePriority: false });
    expect(JSON.stringify(disabled)).toBe(JSON.stringify(absent));
    expect(JSON.stringify(project(canonical, { currentExchangePriority: true })))
      .toBe(JSON.stringify(project(canonical)));
  });

  it('ages the previous protected result after a tool-only response without loosening older history', () => {
    const canonical = history();
    const { prune } = setup(enabled);
    const messages = [...canonical];
    const first = prune({ messages, canonicalMessages: canonical, canonicalPrefixStable: true });
    const old = messages[2].content;
    expect(messages[4].content).toBe(canonical[4].content);
    canonical.push(call('next'), output('next'));
    messages.push(canonical[5], canonical[6]);
    const second = prune({ messages, canonicalMessages: canonical, canonicalPrefixStable: true });
    expect((messages[4].content as string).length).toBeLessThan(300);
    expect(messages[6].content).toBe(canonical[6].content);
    expect(messages[2].content).toBe(old);
    expect(second.fadingTier).toEqual(first.fadingTier);
    const stable = JSON.stringify(second.context);
    expect(JSON.stringify(prune({ messages, canonicalMessages: canonical, canonicalPrefixStable: true }).context)).toBe(stable);
  });

  it('preserves a complete parallel exchange and current bounded call arguments', () => {
    const parent = new AIMessage({ content: '', tool_calls: [
      { id: 'a', name: 'read_file', args: { path: 'a', detail: 'z'.repeat(1000) } },
      { id: 'b', name: 'read_file', args: { path: 'b' } },
    ] });
    const canonical = [new HumanMessage('task'), parent, output('b'), output('a')];
    const result = project(canonical, enabled);
    expect((result.messages[1] as AIMessage).tool_calls).toEqual(parent.tool_calls);
    expect(result.messages[2].content).toBe(canonical[2].content);
    expect(result.messages[3].content).toBe(canonical[3].content);
  });

  it.each([
    { maxTokens: 1200 }, { maxTokens: 16000, instructionTokens: 14900 },
    { maxTokens: 2200, calibrationRatio: 5 },
  ])('uses the unchanged fallback when the full candidate cannot fit: %j', (overrides) => {
    const canonical = history();
    const result = project(canonical, enabled, overrides);
    const baseline = project(canonical, { enabled: true }, overrides);
    expect(JSON.stringify(result.context)).toBe(JSON.stringify(baseline.context));
    expect(result.indexTokenCountMap).toEqual(baseline.indexTokenCountMap);
  });

  it('does not change summarization behavior', () => {
    const canonical = history();
    expect(JSON.stringify(project(canonical, enabled, { summarize: true })))
      .toBe(JSON.stringify(project(canonical, { enabled: true }, { summarize: true })));
  });

  it('derives priority after a newer user turn and a reconstructed pruner', () => {
    const canonical = history();
    canonical.push(new HumanMessage('new task'), call('new-turn'), output('new-turn'));
    const result = project(canonical, enabled);
    const rebuilt = project(canonical, enabled);
    expect(result.messages.at(-1)?.content).toBe(canonical.at(-1)?.content);
    expect((result.messages[4].content as string).length).toBeLessThan(300);
    expect(JSON.stringify(rebuilt.context)).toBe(JSON.stringify(result.context));
  });

  it('preserves Unicode text and opaque reasoning without mutating canonical arrays', () => {
    const parent = call('unicode');
    parent.content = [{ type: 'reasoning', encrypted_content: 'opaque'.repeat(200), summary: [] }];
    const canonical = [new HumanMessage('task'), parent, output('unicode', '🧪é'.repeat(1000))];
    const before = JSON.stringify(canonical);
    const result = project(canonical, enabled);
    expect(result.messages[2].content).toBe(canonical[2].content);
    expect(JSON.stringify(canonical)).toBe(before);
  });

  it.each([
    () => [new HumanMessage('task'), call('open')],
    () => [new HumanMessage('task'), call('a'), output('other')],
    () => [new HumanMessage('task'), call('a'), output('a'), output('a')],
    () => [new HumanMessage('task'), call('a'), output('a'), call('a'), output('a')],
    () => [new HumanMessage('task'), new AIMessage({ content: '', tool_calls: [
      { id: 'a', name: 'read_file', args: {} }, { id: 'a', name: 'read_file', args: {} },
    ] }), output('a')],
    () => [new HumanMessage('task'), call('a'), new ToolMessage({ tool_call_id: 'a', content: [{ type: 'image_url', image_url: 'data:image/png;base64,AA==' }] })],
    () => [new HumanMessage('task'), call('a'), new HumanMessage([{ type: 'tool_result', tool_use_id: 'a', content: text('native') }])],
    () => [new HumanMessage('task'), call('a'), new SystemMessage('interleaved'), output('a')],
    () => [new HumanMessage('task'), call('a'), new HumanMessage('steering'), output('a')],
  ])('leaves incomplete, ambiguous and unsupported exchanges to the existing policy', (make) => {
    const canonical = make();
    expect(JSON.stringify(project(canonical, enabled).context))
      .toBe(JSON.stringify(project(canonical, { enabled: true }).context));
  });

  it('rejects merged source provenance', () => {
    const canonical = history();
    setProviderMessageProvenance(canonical[3], [
      { attribution: 'model', sourceMessageId: 'a' }, { attribution: 'model', sourceMessageId: 'b' },
    ]);
    expect(JSON.stringify(project(canonical, enabled).context))
      .toBe(JSON.stringify(project(canonical, { enabled: true }).context));
  });

  it.each(['chat', 'responses'] as const)('accepts matching %s mirrors without changing call IDs', (format) => {
    const canonical = history();
    const parent = canonical[3] as AIMessage;
    if (format === 'chat') parent.additional_kwargs.tool_calls = [{
      id: 'fresh', type: 'function', function: { name: 'read_file', arguments: '{"path":"fresh"}' },
    }];
    else parent.response_metadata.output = [{ type: 'reasoning', encrypted_content: 'opaque' },
      { type: 'function_call', call_id: 'fresh', name: 'read_file', arguments: '{"path":"fresh"}' }];
    const result = project(canonical, enabled);
    expect(result.messages[4].content).toBe(canonical[4].content);
    expect((result.messages[3] as AIMessage).tool_calls?.[0].id).toBe('fresh');
  });

  it.each(['missing-native-call', 'duplicate-native-call', 'duplicate-chat-call', 'invalid-native-item'] as const)(
    'rejects the inconsistent %s mirror', (format) => {
      const canonical = history();
      const parent = canonical[3] as AIMessage;
      const native = { type: 'function_call', call_id: 'fresh', name: 'read_file', arguments: '{}' };
      if (format === 'missing-native-call') parent.response_metadata.output = [{ type: 'message', content: [] }];
      else if (format === 'duplicate-native-call') parent.response_metadata.output = [native, native];
      else if (format === 'invalid-native-item') parent.response_metadata.output = [null];
      else parent.additional_kwargs.tool_calls = [
        { id: 'fresh', type: 'function', function: { name: 'read_file', arguments: '{}' } },
        { id: 'fresh', type: 'function', function: { name: 'read_file', arguments: '{}' } },
      ];
      expect(JSON.stringify(project(canonical, enabled).context))
        .toBe(JSON.stringify(project(canonical, { enabled: true }).context));
    }
  );
});

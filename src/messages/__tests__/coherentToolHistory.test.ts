import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ContextPruningConfig } from '@/types/graph';
import type { TokenCounter } from '@/types/run';
import { setProviderMessageProvenance } from '@/messages/provenance';
import { createPruneMessages, projectToolMessagesForProvider } from '@/messages/prune';
import { Providers } from '@/common';

const counter: TokenCounter = (message) => Math.ceil((JSON.stringify(message.content).length
  + (message.getType() === 'ai' ? JSON.stringify((message as AIMessage).tool_calls).length : 0)) / 4) + 4;
const enabled = { enabled: true, coherentToolHistory: true };
const call = (id: string, size = 6000, content = ''): AIMessage => new AIMessage({
  content, tool_calls: [{ id, name: 'record', args: { command: id + ':' + 'x'.repeat(size) } }],
});
const output = (id: string, content = 'recorded'): ToolMessage => new ToolMessage({ tool_call_id: id, content });
const history = (): BaseMessage[] => [new SystemMessage('system requirements'), new HumanMessage('TASK: retain this'),
  call('old'), output('old'), new AIMessage('Keep this explanation.'), call('fresh', 200), output('fresh')];

function setup(config: ContextPruningConfig = enabled, options: {
  maxTokens?: number; instructionTokens?: number; calibrationRatio?: number;
  summarize?: boolean; counts?: Record<string, number | undefined>;
} = {}) {
  const prune = createPruneMessages({ maxTokens: options.maxTokens ?? 16000,
    startIndex: 0, tokenCounter: counter, indexTokenCountMap: options.counts ?? {},
    contextPruningConfig: config, getInstructionTokens: () => options.instructionTokens ?? 100,
    calibrationRatio: options.calibrationRatio, summarizationEnabled: options.summarize,
    maxToolResultChars: 20000, fadingTier: { v: 2, budgetTokens: 170, masked: true, latched: true },
  });
  return prune;
}

function project(canonical: BaseMessage[], config: ContextPruningConfig = enabled, options?: Parameters<typeof setup>[1]) {
  const messages = [...canonical];
  return { ...setup(config, options)({ messages, canonicalMessages: canonical }), messages };
}

function ids(messages: BaseMessage[]): string[] {
  return messages.flatMap((message) => {
    if (message.getType() === 'ai') return (message as AIMessage).tool_calls?.map((item) => item.id ?? '') ?? [];
    if (message.getType() === 'tool') return [(message as ToolMessage).tool_call_id];
    return [];
  });
}

describe('coherent completed tool history', () => {
  it('retains exact old and fresh arguments above the fading cap and leaves canonical history unchanged', () => {
    const canonical = history();
    const before = JSON.stringify(canonical);
    const baseline = project(canonical, { enabled: true });
    const result = project(canonical);
    expect((baseline.messages[2] as AIMessage).tool_calls).not.toEqual((canonical[2] as AIMessage).tool_calls);
    expect((result.messages[2] as AIMessage).tool_calls).toEqual((canonical[2] as AIMessage).tool_calls);
    expect(JSON.stringify(canonical)).toBe(before);
    expect(ids(result.context)).toEqual(['old', 'old', 'fresh', 'fresh']);
    const wire = projectToolMessagesForProvider(result.context, 300, Providers.OPENAI, result.exactToolInputIndices);
    expect(ids(wire)).toEqual(ids(result.context));
    expect((wire[2] as AIMessage).tool_calls).toEqual((canonical[2] as AIMessage).tool_calls);
    expect(JSON.stringify(result.context)).not.toContain('_inputPrefix');
  });

  it('keeps an explicitly aliased canonical array immutable while fading a separate projection', () => {
    const canonical = history();
    (canonical[3] as ToolMessage).content = 'old-output:' + 'x'.repeat(25000);
    const before = JSON.stringify(canonical);
    const result = setup()({ messages: canonical, canonicalMessages: canonical });
    expect(JSON.stringify(canonical)).toBe(before);
    expect(result.context[3].content).not.toBe(canonical[3].content);
    expect(result.exactToolInputIndices).toBeDefined();
  });

  it('evicts an old batch within one task while retaining system, task, prose and current results', () => {
    const canonical = history();
    const before = JSON.stringify(canonical);
    const result = project(canonical, enabled, { maxTokens: 1200 });
    expect(ids(result.context)).toEqual(['fresh', 'fresh']);
    for (const index of [0, 1, 4]) expect(result.context).toContain(canonical[index]);
    expect(result.context.at(-1)?.content).toBe(canonical.at(-1)?.content);
    expect(result.remainingContextTokens).toBeGreaterThanOrEqual(0);
    expect(result.indexTokenCountMap[2]).toBeGreaterThan(1000);
    expect(JSON.stringify(canonical)).toBe(before);
    expect(result.exactToolInputIndices).toEqual(new Set([3]));
  });

  it('evicts a closed parallel batch atomically even with reversed result order', () => {
    const parent = call('old');
    parent.tool_calls!.push({ id: 'parallel', name: 'record', args: { command: 'second' } });
    const canonical = [new HumanMessage('task'), parent, output('parallel'), output('old'), call('fresh', 200), output('fresh')];
    expect(ids(project(canonical, enabled, { maxTokens: 1200 }).context)).toEqual(['fresh', 'fresh']);
  });

  it('evicts earlier history to retain a full fresh result that would otherwise be faded', () => {
    const canonical = [new HumanMessage('task'), call('old', 4000), output('old'), call('fresh', 200), output('fresh', 'FRESH:' + 'y'.repeat(4000))];
    const result = project(canonical, enabled, { maxTokens: 1600 });
    expect(ids(result.context)).toEqual(['fresh', 'fresh']);
    expect(result.context.at(-1)?.content).toBe(canonical.at(-1)?.content);
    expect(result.remainingContextTokens).toBeGreaterThanOrEqual(0);
  });

  it('preserves pending terminal calls while evicting an already consumed exchange', () => {
    const canonical = [new HumanMessage('task'), call('old'), output('old'), call('pending', 200)];
    const result = project(canonical, enabled, { maxTokens: 1200 });
    expect(ids(result.context)).toEqual(['pending']);
    expect(result.context.at(-1)).toBe(canonical.at(-1));
    expect(result.exactToolInputIndices).toEqual(new Set([1]));
  });

  it('keeps an irreducible current exchange exact and reports over-budget state', () => {
    const canonical = [new HumanMessage('task'), call('fresh'), output('fresh')];
    const result = project(canonical, enabled, { maxTokens: 1200 });
    expect(ids(result.context)).toEqual(['fresh', 'fresh']);
    expect(result.remainingContextTokens).toBeLessThan(0);
    expect((result.context[1] as AIMessage).tool_calls).toEqual((canonical[1] as AIMessage).tool_calls);
  });

  it('never evicts assistant prose bundled with a closed call', () => {
    const canonical = [new HumanMessage('task'), call('old', 6000, 'A necessary explanation'), output('old'), call('fresh', 200), output('fresh')];
    const result = project(canonical, enabled, { maxTokens: 1200 });
    expect(ids(result.context)).toEqual(['old', 'old', 'fresh', 'fresh']);
    expect(result.remainingContextTokens).toBeLessThan(0);
  });

  it('counts exact arguments despite a small cached host estimate and preserves a larger estimate', () => {
    const canonical = history();
    const small = project(canonical, enabled, { maxTokens: 1200, counts: { 2: 1 } });
    expect(ids(small.context)).toEqual(['fresh', 'fresh']);
    expect(small.indexTokenCountMap[2]).toBe(counter(canonical[2]));
    const large = project(canonical, enabled, { counts: { 2: 5000 } });
    expect(large.indexTokenCountMap[2]).toBe(5000);
  });

  it('includes calibrated instruction, reply and output headroom in the returned remainder', () => {
    const canonical = history();
    const result = project(canonical, enabled, { maxTokens: 4000, instructionTokens: 700, calibrationRatio: 2 });
    const raw = result.context.reduce((total, message) => total + result.indexTokenCountMap[result.messages.indexOf(message)]!, 0);
    expect(result.contextBudget).toBe(3800);
    expect(result.remainingContextTokens).toBe(3800 - Math.round(raw * 2) - 6 - 700);
    expect(ids(result.context)).toEqual(['fresh', 'fresh']);
  });

  it('calibrates the next turn against retained original indices rather than evicted calls', () => {
    const canonical = history();
    const messages = [...canonical];
    const prune = setup(enabled, { maxTokens: 1200 });
    const first = prune({ messages, canonicalMessages: canonical });
    const sent = first.context.reduce((total, message) => total + first.indexTokenCountMap[messages.indexOf(message)]!, 0);
    canonical.push(call('next', 100), output('next'));
    messages.push(canonical[7], canonical[8]);
    const second = prune({ messages, canonicalMessages: canonical, usageMetadata: { input_tokens: sent + 100, output_tokens: 1 } });
    expect(second.calibrationRatio).toBe(1);
    expect(ids(second.context)).not.toContain('old');
  });

  it('keeps the eviction watermark after pressure falls and resets it on prefix replacement', () => {
    let instructionTokens = 1100;
    const canonical = history();
    const messages = [...canonical];
    const prune = createPruneMessages({ maxTokens: 2500, startIndex: 0, tokenCounter: counter,
      indexTokenCountMap: {}, contextPruningConfig: enabled, getInstructionTokens: () => instructionTokens });
    expect(ids(prune({ messages, canonicalMessages: canonical }).context)).not.toContain('old');
    instructionTokens = 0;
    expect(ids(prune({ messages, canonicalMessages: canonical }).context)).not.toContain('old');
    const replacement = history();
    expect(ids(prune({ messages: [...replacement], canonicalMessages: replacement }).context)).toContain('old');
  });

  it('recounts a smaller replacement prefix rather than retaining the former large argument cost', () => {
    const prune = setup(enabled, { maxTokens: 1200 });
    const first = history();
    expect(ids(prune({ messages: [...first], canonicalMessages: first }).context)).not.toContain('old');
    const replacement = [new HumanMessage('replacement'), call('small', 100), output('small'), call('fresh', 100), output('fresh')];
    const result = prune({ messages: [...replacement], canonicalMessages: replacement });
    expect(ids(result.context)).toEqual(['small', 'small', 'fresh', 'fresh']);
    expect(result.remainingContextTokens).toBeGreaterThan(0);
    expect(result.indexTokenCountMap[1]).toBe(counter(replacement[1]));
  });

  it('admits ordinary text-array content and a verified complete Responses item-ID map', () => {
    const canonical = history();
    const parent = canonical[2] as AIMessage;
    parent.content = [];
    parent.additional_kwargs.__openai_function_call_ids__ = { old: 'fc_old' };
    parent.response_metadata.output = [{ type: 'function_call', id: 'fc_old', call_id: 'old', name: 'record', arguments: JSON.stringify(parent.tool_calls![0].args) }];
    expect(project(canonical).exactToolInputIndices).toBeDefined();
  });

  it.each(['chat', 'responses'] as const)('admits exact %s mirrors without changing them', (format) => {
    const canonical = history();
    const parent = canonical[2] as AIMessage;
    const args = JSON.stringify(parent.tool_calls![0].args);
    if (format === 'chat') parent.additional_kwargs.tool_calls = [{ id: 'old', type: 'function', function: { name: 'record', arguments: args } }];
    else parent.response_metadata.output = [{ type: 'function_call', call_id: 'old', name: 'record', arguments: args }];
    const before = JSON.stringify(parent);
    const result = project(canonical);
    expect(result.exactToolInputIndices).toBeDefined();
    expect(JSON.stringify(result.messages[2])).toBe(before);
  });

  it('leaves reordered native mirrors to legacy instead of claiming a coherent ordinary batch', () => {
    const canonical = history();
    const parent = canonical[2] as AIMessage;
    parent.tool_calls!.push({ id: 'parallel', name: 'record', args: { command: 'second' } });
    canonical.splice(4, 0, output('parallel'));
    parent.response_metadata.output = [...parent.tool_calls!].reverse().map((item) => ({
      type: 'function_call', call_id: item.id, name: item.name, arguments: JSON.stringify(item.args),
    }));
    expect(project(canonical).exactToolInputIndices).toBeUndefined();
    expect(JSON.stringify(project(canonical))).toBe(JSON.stringify(project(canonical, { enabled: true })));
  });

  it.each([
    () => [new HumanMessage('task'), call('a'), output('a'), call('a'), output('a')],
    () => [new HumanMessage('task'), call('a'), output('a'), output('a')],
    () => [new HumanMessage('task'), call('a'), output('other')],
    () => [new HumanMessage('task'), call('a'), new SystemMessage('interleaved'), output('a')],
    () => [new HumanMessage('task'), call('a'), new HumanMessage('steer'), output('a')],
    () => [new HumanMessage('task'), call('a'), new HumanMessage([{ type: 'tool_result', tool_use_id: 'a', content: 'native' }])],
    () => [new HumanMessage('task'), call('a'), new ToolMessage({ tool_call_id: 'a', content: [{ type: 'image_url', image_url: 'data:image/png;base64,AA==' }] })],
    () => [new HumanMessage('task'), call('oversize', 210000), output('oversize')],
    () => { const parent = call('a'); parent.tool_calls!.push(parent.tool_calls![0]); return [parent, output('a')]; },
    () => { const parent = call('a'); parent.tool_calls!.push({ id: 'b', name: 'record', args: {} }); return [parent, output('a')]; },
    () => { const parent = call('a'); parent.additional_kwargs.tool_calls = [{ id: 'a', type: 'function', function: { name: 'wrong', arguments: JSON.stringify(parent.tool_calls![0].args) } }]; return [parent, output('a')]; },
    () => { const parent = call('a'); parent.response_metadata.output = [{ type: 'reasoning', encrypted_content: 'opaque' }]; return [parent, output('a')]; },
    () => { const parent = call('a'); parent.additional_kwargs.reasoning_content = 'provider reasoning'; return [parent, output('a')]; },
    () => { const parent = call('a'); parent.additional_kwargs.tool_outputs = [{ type: 'function_call', call_id: 'a', name: 'record', arguments: '{}' }]; return [parent, output('a')]; },
    () => { const parent = call('a'); parent.additional_kwargs.reasoning = { encrypted_content: 'opaque' }; return [parent, output('a')]; },
    () => { const parent = call('a'); parent.response_metadata.preempted = true; return [parent, output('a')]; },
    () => { const parent = call('a'); parent.additional_kwargs.__openai_function_call_ids__ = { a: 'wrong' }; return [parent, output('a')]; },
    () => { const result = output('a'); setProviderMessageProvenance(result, [{ attribution: 'model', sourceMessageId: 'x' }, { attribution: 'model', sourceMessageId: 'y' }]); return [call('a'), result]; },
    () => { const parent = call('a'); const result = output('a'); for (const message of [parent, result]) setProviderMessageProvenance(message, [{ attribution: 'model', sourceMessageId: 'same' }]); return [parent, result]; },
  ])('leaves unsupported and ambiguous histories byte-equivalent to legacy', (make) => {
    const canonical = make();
    const result = project(canonical);
    const baseline = project(canonical, { enabled: true });
    expect(JSON.stringify(result)).toBe(JSON.stringify(baseline));
    expect(result.exactToolInputIndices).toBeUndefined();
  });

  it('keeps disabled, missing-enabled and summary-on behavior unchanged', () => {
    const canonical = history();
    expect(JSON.stringify(project(canonical, { enabled: true, coherentToolHistory: false })))
      .toBe(JSON.stringify(project(canonical, { enabled: true })));
    expect(JSON.stringify(project(canonical, { coherentToolHistory: true })))
      .toBe(JSON.stringify(project(canonical, {})));
    expect(JSON.stringify(project(canonical, enabled, { summarize: true })))
      .toBe(JSON.stringify(project(canonical, { enabled: true }, { summarize: true })));
  });

  it.each([
    ['missing result', (messages: BaseMessage[]) => { messages.splice(3, 1); }],
    ['extra result', (messages: BaseMessage[]) => { messages.splice(4, 0, output('old')); }],
    ['duplicate parent ID', (messages: BaseMessage[]) => {
      (messages[5] as AIMessage).tool_calls![0].id = 'old';
      (messages[6] as ToolMessage).tool_call_id = 'old';
    }],
    ['native human result', (messages: BaseMessage[]) => {
      messages[3] = new HumanMessage([{ type: 'tool_result', tool_use_id: 'old', content: 'native' }]);
    }],
    ['mismatched mirror', (messages: BaseMessage[]) => {
      messages[2].additional_kwargs.tool_calls = [{ id: 'old', type: 'function', function: { name: 'record', arguments: '{}' } }];
    }],
    ['merged provenance', (messages: BaseMessage[]) => {
      setProviderMessageProvenance(messages[3], [{ attribution: 'model', sourceMessageId: 'x' }, { attribution: 'model', sourceMessageId: 'y' }]);
    }],
    ['missing arguments', (messages: BaseMessage[]) => { Reflect.deleteProperty((messages[2] as AIMessage).tool_calls![0], 'args'); }],
  ] as const)('uses byte-identical legacy behavior under eviction pressure for %s', (_name, invalidate) => {
    const canonical = history();
    invalidate(canonical);
    const before = JSON.stringify(canonical);
    const result = project(canonical, enabled, { maxTokens: 1200 });
    const baseline = project(canonical, { enabled: true }, { maxTokens: 1200 });
    expect(result.exactToolInputIndices).toBeUndefined();
    expect(JSON.stringify(result)).toBe(JSON.stringify(baseline));
    expect(JSON.stringify(canonical)).toBe(before);
  });
});

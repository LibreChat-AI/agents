import { Runnable } from '@langchain/core/runnables';
import { AIMessage, AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { StandardGraph } from '@/graphs/Graph';
import { Providers } from '@/common';

class RecordingModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  calls: BaseMessage[][] = [];
  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.calls.push(messages);
    return new AIMessageChunk('answer');
  }
}

const count: t.TokenCounter = () => 10;

function makeGraph(mode: 'normal' | 'manual' | 'trigger'): StandardGraph {
  return new StandardGraph({
    runId: mode, tokenCounter: count,
    agents: [{
      agentId: 'agent', provider: Providers.OPENAI,
      instructions: 'Same system prompt', maxContextTokens: 1_000,
      summarizeOnly: mode === 'manual', summarizationEnabled: true,
      summarizationConfig: {trigger: {type: 'messages_to_refine', value: mode === 'trigger' ? 1 : 10_000}},
      clientOptions: {disableStreaming: true},
    }],
  });
}

it('manual and automatic requests carry the normal turn’s pruned window, not 720 stored messages', async () => {
  const messages: BaseMessage[] = [];
  for (let i = 0; i < 360; i++) messages.push(new HumanMessage(`q${i}`), new AIMessage(`a${i}`));
  const normal = makeGraph('normal');
  const model = new RecordingModel();
  normal.overrideModel = model;
  const config = {metadata: {langgraph_node: 'agent=agent', langgraph_step: 1}, configurable: {thread_id: 'window'}};
  await normal.createCallModel('agent')({messages}, config);
  expect(model.calls).toHaveLength(1);
  const pruned = model.calls[0].filter(message => message.getType() !== 'system');
  expect(pruned.length).toBeLessThan(720);
  expect(pruned.length).toBeGreaterThan(0);
  for (const mode of ['manual', 'trigger'] as const) {
    const graph = makeGraph(mode);
    const result = await graph.createCallModel('agent')({messages}, config);
    expect(result.summarizationRequest?.windowMessages).toEqual(pruned);
  }
});

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, isAIMessage } from '@langchain/core/messages';
import { END, START, Command, Annotation, StateGraph, MemorySaver, MessagesAnnotation } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { ToolNode } from '@/tools/ToolNode';
import { HookRegistry } from '@/hooks';
import { Providers } from '@/common';
import { Run } from '@/run';

const State = Annotation.Root({
  ...MessagesAnnotation.spec,
  runStepState: Annotation<t.RunStepResumeState | undefined>({ reducer: (current, update) => update ?? current, default: () => undefined }),
});

describe('pending checkpoint proposal identity', () => {
  test.each(['proposal', 'principal', 'scope'] as const)('fails closed on an unaddressed %s change', async (change) => {
    let ask = true;
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', { hooks: [async () => ({ decision: ask ? 'ask' as const : 'allow' as const })] });
    const left = jest.fn(async () => 'left effect');
    const right = jest.fn(async () => 'right effect');
    const node = (name: 'left' | 'right') => new ToolNode<t.IState>({
      agentId: name, hookRegistry: hooks, humanInTheLoop: { enabled: true }, directToolNames: new Set([name]),
      tools: [tool(name === 'left' ? left : right, { name, description: 'Review', schema: z.object({ value: z.string() }) })],
    });
    const leftNode = node('left');
    const rightNode = node('right');
    const invoke = (toolNode: ToolNode<t.IState>, name: string) => async (state: typeof State.State, config: RunnableConfig): Promise<t.IState> => {
      const message = state.messages.findLast(isAIMessage);
      if (message == null) throw new Error('Missing checkpoint proposal');
      return toolNode.invoke({ messages: [new AIMessage({ id: message.id, content: '', tool_calls: (message.tool_calls ?? []).filter((call) => call.name === name) })] }, config);
    };
    const checkpointer = new MemorySaver();
    const workflow = new StateGraph(State)
      .addNode('left', invoke(leftNode, 'left')).addNode('right', invoke(rightNode, 'right'))
      .addEdge(START, 'left').addEdge(START, 'right').addEdge('left', END).addEdge('right', END)
      .compile({ checkpointer });
    const run = await Run.create<t.IState>({
      runId: 'pending-proposal-identity', hooks, humanInTheLoop: { enabled: true }, skipCleanup: true,
      graphConfig: { type: 'multi-agent', entryAgentId: 'left', edges: [], compileOptions: { checkpointer }, agents: ['left', 'right'].map((agentId) => ({
        agentId, provider: Providers.OPENAI, clientOptions: { modelName: 'unused', apiKey: 'unused' },
      })) },
    });
    run.graphRunnable = workflow;
    const config = { version: 'v2' as const, configurable: { thread_id: 'pending-proposal', user_id: 'original', __librechat_tool_approval_execution_scope: 'original' } };
    const proposal = (value: string) => new AIMessage({ id: 'stable-proposal', content: '', tool_calls: [
      { id: 'left-call', name: 'left', args: { value: 'left' } }, { id: 'right-call', name: 'right', args: { value } },
    ] });
    await run.processStream({ messages: [proposal('right')] }, config);
    const first = run.getInterrupt()?.pendingInterrupts?.find((entry) => isToolApprovalInterrupt(entry.payload) && entry.payload.action_requests[0].name === 'left');
    if (first == null) throw new Error('Missing first approval');
    ask = false;
    await run.resume({ [first.interruptId]: [{ type: 'approve' }] }, config);
    expect(left.mock.calls.length).toBe(1);
    expect(right.mock.calls.length).toBe(0);
    expect(run.getInterrupt()?.pendingInterrupts).toHaveLength(1);
    const changedConfig = { ...config, configurable: { ...config.configurable,
      user_id: change === 'principal' ? 'changed' : 'original', __librechat_tool_approval_execution_scope: change === 'scope' ? 'changed' : 'original',
    } };
    const failure = await run.processStream(new Command({ update: { messages: [change === 'proposal' ? proposal('changed') : new HumanMessage('Still pending')] } }), changedConfig)
      .then(() => undefined, (error: Error) => error);
    const errors = failure instanceof AggregateError ? failure.errors : [failure];
    expect(errors.some((error) => error instanceof Error && error.message.includes(change === 'proposal' ? 'proposal changed' : 'execution owner changed'))).toBe(true);
    for (const error of errors) expect(error instanceof Error && /proposal changed|execution owner changed/.test(error.message)).toBe(true);
    expect(left.mock.calls.length).toBe(1);
    expect(right.mock.calls.length).toBe(0);
  });
});

import { randomUUID } from 'node:crypto';
import { createWorkerModelLifecycle } from './admission';

it('does not enroll summary or unrelated models into the starting graph admission', async () => {
  const call = jest.fn();
  const worker = createWorkerModelLifecycle('root', 16, { call });
  const nodeId = randomUUID();
  const callback = worker.callback;
  await callback.handleChainStart?.(
    { id: [], lc: 1, type: 'not_implemented' },
    {},
    nodeId,
    undefined,
    undefined,
    { langgraph_node: 'agent=root' },
    undefined,
    'agent=root'
  );
  for (const metadata of [
    { agentId: 'other', langgraph_node: 'agent=root' },
    { agentId: 'root', langgraph_node: 'agent=root', summarization: true },
  ]) {
    await callback.handleChatModelStart?.(
      { id: [], lc: 1, type: 'not_implemented' },
      [],
      randomUUID(),
      nodeId,
      undefined,
      undefined,
      metadata
    );
  }
  await callback.handleChainEnd?.({ messages: [{}] }, nodeId);
  expect(call).not.toHaveBeenCalled();
  expect(() => worker.current()).toThrow();
});

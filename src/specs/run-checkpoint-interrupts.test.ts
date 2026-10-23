import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
  END,
  START,
  Command,
  Annotation,
  StateGraph,
  MemorySaver,
  MessagesAnnotation,
  interrupt,
} from '@langchain/langgraph';
import type { StateSnapshot } from '@langchain/langgraph';
import type * as t from '@/types';
import { Providers } from '@/common';
import { Run } from '@/run';

const State = Annotation.Root({
  ...MessagesAnnotation.spec,
  runStepState: Annotation<t.RunStepResumeState | undefined>({
    reducer: (current, update) => update ?? current,
    default: () => undefined,
  }),
});

type Question = { question: string };
type Exposure = 'propagated' | 'nested-only';

function exposeNestedOnly(snapshot: StateSnapshot): StateSnapshot {
  return {
    ...snapshot,
    tasks: snapshot.tasks.map((task) => {
      const state = task.state;
      if (state == null || !('tasks' in state)) return task;
      return { ...task, interrupts: [], state: exposeNestedOnly(state) };
    }),
  };
}

function harness(exposure: Exposure) {
  const checkpointer = new MemorySaver();
  const completed: string[] = [];
  const config = {
    configurable: { thread_id: `mixed-depth-${exposure}` },
    version: 'v2' as const,
  };
  const ask = (question: string) => () => {
    const answer = interrupt<Question, string>({ question });
    completed.push(`${question}:${answer}`);
    return { messages: [new AIMessage(`${question}:${answer}`)] };
  };
  const build = async () => {
    const leaf = new StateGraph(State)
      .addNode('leaf', ask('leaf'))
      .addEdge(START, 'leaf')
      .addEdge('leaf', END)
      .compile();
    const branch = new StateGraph(State)
      .addNode('middle', ask('middle'))
      .addNode('deep', leaf)
      .addEdge(START, 'middle')
      .addEdge(START, 'deep')
      .addEdge('middle', END)
      .addEdge('deep', END)
      .compile();
    const workflow = new StateGraph(State)
      .addNode('branch', branch)
      .addNode('root', ask('root'))
      .addEdge(START, 'root')
      .addEdge(START, 'branch')
      .addEdge('root', END)
      .addEdge('branch', END)
      .compile({ checkpointer });
    if (exposure === 'nested-only') {
      const getState = workflow.getState.bind(workflow);
      jest
        .spyOn(workflow, 'getState')
        .mockImplementation(async (...args) =>
          exposeNestedOnly(await getState(...args))
        );
    }
    const run = await Run.create<t.IState>({
      runId: 'mixed-depth-checkpoint',
      skipCleanup: true,
      graphConfig: {
        type: 'multi-agent',
        entryAgentId: 'root',
        edges: [],
        agents: ['root', 'middle', 'leaf'].map((agentId) => ({
          agentId,
          provider: Providers.OPENAI,
          clientOptions: { apiKey: 'unused', modelName: 'unused' },
        })),
        compileOptions: { checkpointer },
      },
    });
    run.graphRunnable = workflow;
    return { run, workflow };
  };
  return { build, completed, config };
}

function checkpointQuestions(
  snapshot: StateSnapshot
): NonNullable<t.RunInterruptResult<Question>['pendingInterrupts']> {
  const states = [snapshot];
  for (const name of ['branch', 'deep']) {
    const state = states
      .at(-1)
      ?.tasks.find((task) => task.name === name)?.state;
    if (state == null || !('tasks' in state))
      throw new Error('Missing nested state');
    states.push(state);
  }
  return states.flatMap((state) =>
    state.tasks
      .filter((task) => ['root', 'middle', 'leaf'].includes(task.name))
      .flatMap((task) => task.interrupts)
      .map((entry) => {
        if (entry.id == null) throw new Error('Missing interrupt ID');
        return { interruptId: entry.id, payload: entry.value as Question };
      })
  );
}

function decisionsFor(
  pending: NonNullable<t.RunInterruptResult<Question>['pendingInterrupts']>,
  addressed: string
): Record<string, string> {
  return Object.fromEntries(
    pending
      .filter(
        (entry) => addressed === 'all' || entry.payload.question === addressed
      )
      .map((entry) => [entry.interruptId, `${entry.payload.question}-answer`])
  );
}

describe.each(['propagated', 'nested-only'] as const)(
  'mixed-depth checkpoint interrupts: %s',
  (exposure) => {
    test.each(['all', 'root', 'middle', 'leaf'])(
      'discovers checkpoint IDs and routes %s resume maps after rebuild',
      async (addressed) => {
        const h = harness(exposure);
        const { workflow: initial } = await h.build();
        await initial.invoke(
          { messages: [new HumanMessage('Review')] },
          h.config
        );
        const pending = checkpointQuestions(
          await initial.getState(h.config, { subgraphs: true })
        );
        expect(pending).toHaveLength(3);
        const { run: rebuilt, workflow } = await h.build();
        const snapshot = await workflow.getState(h.config, { subgraphs: true });
        const primary = snapshot.tasks.flatMap((task) => task.interrupts)[0];
        expect(primary).toBeDefined();
        let restored: t.RunInterruptResult<Question> | undefined;
        const processStream = rebuilt.processStream.bind(rebuilt);
        jest.spyOn(rebuilt, 'processStream').mockImplementation((...args) => {
          restored = rebuilt.getInterrupt<Question>();
          return processStream(...args);
        });
        await rebuilt.resume(decisionsFor(pending, addressed), h.config);
        expect(restored?.interruptId).toBe(primary.id);
        expect(restored?.pendingInterrupts).toHaveLength(3);
        expect(
          restored?.pendingInterrupts
            ?.map((entry) => entry.payload.question)
            .sort()
        ).toEqual(['leaf', 'middle', 'root']);
        expect(
          new Set(
            restored?.pendingInterrupts?.map((entry) => entry.interruptId)
          )
        ).toEqual(new Set(pending.map((entry) => entry.interruptId)));
        expect(h.completed.sort()).toEqual(
          ['root', 'middle', 'leaf']
            .filter((question) => addressed === 'all' || question === addressed)
            .map((question) => `${question}:${question}-answer`)
            .sort()
        );
        if (addressed !== 'all') {
          const remaining = pending.filter(
            (entry) => entry.payload.question !== addressed
          );
          const { run: again } = await h.build();
          const decisions = decisionsFor(remaining, 'all');
          jest.spyOn(again, 'processStream').mockImplementation((input) => {
            expect(again.getInterrupt<Question>()?.pendingInterrupts).toEqual(
              expect.arrayContaining(remaining)
            );
            expect(
              again.getInterrupt<Question>()?.pendingInterrupts
            ).toHaveLength(2);
            if (!(input instanceof Command))
              throw new Error('Missing resume Command');
            expect(input.resume).toEqual(decisions);
            return Promise.resolve(undefined);
          });
          await again.resume(decisions, h.config);
          return;
        }
        expect(rebuilt.getInterrupt()).toBeUndefined();
        expect(h.completed.sort()).toEqual([
          'leaf:leaf-answer',
          'middle:middle-answer',
          'root:root-answer',
        ]);
        const { run: finished, workflow: finishedWorkflow } = await h.build();
        const completedState = await finishedWorkflow.getState(h.config, {
          subgraphs: true,
        });
        expect(completedState.next).toEqual([]);
        await finished.resume({}, h.config);
        expect(finished.getInterrupt()).toBeUndefined();
        expect(h.completed).toHaveLength(3);
      }
    );

    test('direct processStream Commands restore and resume every pending depth', async () => {
      const h = harness(exposure);
      const { workflow: initial } = await h.build();
      await initial.invoke(
        { messages: [new HumanMessage('Review')] },
        h.config
      );
      const pending = checkpointQuestions(
        await initial.getState(h.config, { subgraphs: true })
      );
      const { run: rebuilt } = await h.build();
      await rebuilt.processStream(
        new Command({ resume: decisionsFor(pending, 'all') }),
        h.config
      );
      expect(rebuilt.getInterrupt()).toBeUndefined();
      expect(h.completed.sort()).toEqual([
        'leaf:leaf-answer',
        'middle:middle-answer',
        'root:root-answer',
      ]);
    });
  }
);

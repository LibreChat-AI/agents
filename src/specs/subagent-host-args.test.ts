import { HumanMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { Constants, GraphEvents, Providers, ToolEndHandler } from '@/index';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

const CHILD_RESPONSE = 'Reviewed the pull request on the selected machine.';

const callerConfig: Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} = {
  configurable: { thread_id: 'subagent-host-args-thread' },
  streamMode: 'values',
  version: 'v2' as const,
};

const childInputs = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
  maxContextTokens: 8000,
});

const createParentAgent = (
  resolveReviewer: jest.Mock<Promise<t.AgentInputs>, [t.SubagentResolveContext]>
): t.AgentInputs => ({
  agentId: 'parent',
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: 'Delegate reviews with the subagent tool.',
  maxContextTokens: 8000,
  subagentConfigs: [
    {
      type: 'reviewer',
      name: 'PR Reviewer',
      description: 'Reviews pull requests.',
      configId: 'reviewer@v1',
      resolveAgentInputs: resolveReviewer,
      hostArgs: {
        machine: {
          description: 'Code machine the reviewer runs on.',
          enum: ['byom-laptop', 'code-api'],
        },
      },
    },
    {
      type: 'researcher',
      name: 'Researcher',
      description: 'Researches topics.',
      agentInputs: childInputs('researcher'),
    },
  ],
});

async function runWithToolCall(
  resolveReviewer: jest.Mock<
    Promise<t.AgentInputs>,
    [t.SubagentResolveContext]
  >,
  args: Record<string, string>
): Promise<string> {
  const run = await Run.create<t.IState>({
    runId: `subagent-host-args-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      agents: [createParentAgent(resolveReviewer)],
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers: { [GraphEvents.TOOL_END]: new ToolEndHandler() },
  });
  const toolCall: ToolCall = {
    id: 'call_review',
    name: Constants.SUBAGENT,
    args,
    type: 'tool_call',
  };
  run.Graph?.overrideTestModel(['Delegating.', 'Done.'], 10, [toolCall]);
  await run.processStream(
    { messages: [new HumanMessage('Review the PR.')] },
    callerConfig
  );
  const result = run
    .getRunMessages()
    ?.find(
      (message) =>
        message._getType() === 'tool' &&
        'name' in message &&
        message.name === Constants.SUBAGENT
    );
  return String(result?.content ?? '');
}

describe('subagent host arguments through a run', () => {
  jest.setTimeout(30000);

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            constructor(_options: any) {
              super({ responses: [CHILD_RESPONSE] });
            }
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  it('delivers the parent’s choice to the selected resolver', async () => {
    const resolveReviewer = jest.fn(
      async (_context: t.SubagentResolveContext) => childInputs('reviewer')
    );

    const content = await runWithToolCall(resolveReviewer, {
      description: 'Review PR #600.',
      subagent_type: 'reviewer',
      machine: 'byom-laptop',
    });

    expect(content).toContain(CHILD_RESPONSE);
    expect(resolveReviewer).toHaveBeenCalledTimes(1);
    expect(resolveReviewer.mock.calls[0][0].hostArgs).toEqual({
      machine: 'byom-laptop',
    });
  });

  it('falls back to the host default when the parent omits the argument', async () => {
    const resolveReviewer = jest.fn(
      async (_context: t.SubagentResolveContext) => childInputs('reviewer')
    );

    const content = await runWithToolCall(resolveReviewer, {
      description: 'Review PR #600.',
      subagent_type: 'reviewer',
    });

    expect(content).toContain(CHILD_RESPONSE);
    expect(resolveReviewer.mock.calls[0][0].hostArgs).toBeUndefined();
  });

  it('returns a model-visible error for an argument the selected type does not accept', async () => {
    const resolveReviewer = jest.fn(
      async (_context: t.SubagentResolveContext) => childInputs('reviewer')
    );

    const content = await runWithToolCall(resolveReviewer, {
      description: 'Research the topic.',
      subagent_type: 'researcher',
      machine: 'code-api',
    });

    expect(content).toContain(
      'Subagent "researcher" does not accept "machine". Omit it for this subagent type.'
    );
    expect(resolveReviewer).not.toHaveBeenCalled();
  });
});

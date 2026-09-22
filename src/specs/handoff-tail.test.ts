import { DynamicStructuredTool } from '@langchain/core/tools';
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  getBufferString,
} from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { inspectProviderMessageProvenance } from '@/messages/provenance';
import { Constants, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

const HANDOFF_CUE =
  'Continue as the receiving agent using the preceding user request and context.';
const USER_REQUEST = 'Analyze the organization using the fresh data.';
const PREAMBLE = 'Handing off with the organization context and specific asks.';

const createAgent = (
  agentId: string,
  provider: Providers.OPENAI | Providers.ANTHROPIC = Providers.OPENAI,
  model = 'claude-opus-4-8'
): t.AgentInputs => ({
  agentId,
  provider,
  clientOptions: { model, apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
  maxContextTokens: 28000,
});

async function executeHandoff({
  agents = [createAgent('router'), createAgent('recipient')],
  edges = [{ from: 'router', to: 'recipient', edgeType: 'handoff' }],
  toolCalls = [
    { id: 'transfer', name: `${Constants.LC_TRANSFER_TO_}recipient`, args: {} },
  ],
  preamble = PREAMBLE,
  messages = [new HumanMessage(USER_REQUEST)],
}: {
  agents?: t.AgentInputs[];
  edges?: t.GraphEdge[];
  toolCalls?: ToolCall[];
  preamble?: string;
  messages?: BaseMessage[];
} = {}): Promise<{
  requests: BaseMessage[][];
  run: Run<t.BaseGraphState>;
  countedMessages: BaseMessage[];
}> {
  const countedMessages: BaseMessage[] = [];
  const run = await Run.create({
    runId: `handoff-tail-${Math.random()}`,
    graphConfig: { type: 'multi-agent', agents, edges },
    tokenCounter: (message) => {
      countedMessages.push(message);
      return getBufferString([message]).length;
    },
    returnContent: true,
    skipCleanup: true,
  });
  if (run.Graph == null) {
    throw new Error('Expected a multi-agent graph');
  }
  const model = new FakeChatModel({
    responses: [preamble, 'Analysis complete', 'Second analysis complete'],
    toolCalls,
  });
  const streamSpy = jest.spyOn(model, '_streamResponseChunks');
  run.Graph.overrideModel = model;
  await run.processStream(
    { messages },
    { configurable: { thread_id: 'handoff-tail-thread' }, version: 'v2' }
  );
  return {
    requests: streamSpy.mock.calls.map(([messages]) => messages),
    run,
    countedMessages,
  };
}

describe('Handoff message tails', () => {
  it.each([
    [
      'Claude behind an OpenAI-compatible gateway',
      Providers.OPENAI,
      'claude-opus-4-8',
    ],
    ['an opaque gateway model alias', Providers.OPENAI, 'analytics-route'],
    ['a prefill-tolerant model', Providers.OPENAI, 'gpt-4.1'],
    ['native Claude', Providers.ANTHROPIC, 'claude-opus-4-8'],
  ] as const)(
    'grounds an instructionless assistant tail for %s',
    async (_label, provider, model) => {
      const { requests, run, countedMessages } = await executeHandoff({
        agents: [
          createAgent('router'),
          createAgent('recipient', provider, model),
        ],
      });

      expect(requests).toHaveLength(2);
      const received = requests[1];
      expect(received.at(-2)?.getType()).toBe('ai');
      expect(received.at(-2)?.content).toBe(PREAMBLE);
      expect(received.at(-1)?.getType()).toBe('human');
      expect(received.at(-1)?.content).toBe(HANDOFF_CUE);
      expect(
        received.filter((message) => message.content === HANDOFF_CUE)
      ).toHaveLength(1);
      expect(received.some((message) => message.content === USER_REQUEST)).toBe(
        true
      );
      expect(getBufferString(received)).not.toContain(
        `${Constants.LC_TRANSFER_TO_}recipient`
      );
      expect(received.at(-1)?.additional_kwargs).toMatchObject({
        role: 'user',
        isMeta: true,
        source: 'routing',
      });
      expect(inspectProviderMessageProvenance(received.at(-1)!)).toMatchObject({
        status: 'valid',
        provenance: { parts: [{ attribution: 'synthetic' }] },
      });
      expect(
        countedMessages.some((message) => message.content === HANDOFF_CUE)
      ).toBe(true);
      expect(run.getRunMessages()?.at(-1)?.content).toBe('Analysis complete');
    }
  );

  it.each([undefined, '', '  \n ', 'Analyze only the latest usage.'])(
    'adds exactly one user turn when optional instructions are %j',
    async (instructions) => {
      const { requests } = await executeHandoff({
        edges: [
          {
            from: 'router',
            to: 'recipient',
            edgeType: 'handoff',
            prompt: 'Instructions for the receiving agent',
          },
        ],
        toolCalls: [
          {
            id: 'transfer',
            name: `${Constants.LC_TRANSFER_TO_}recipient`,
            args: instructions === undefined ? {} : { instructions },
          },
        ],
      });
      const received = requests[1];
      expect(
        received.filter((message) => message.getType() === 'human')
      ).toHaveLength(2);
      const trimmedInstructions = instructions?.trim() ?? '';
      expect(received.at(-1)?.content).toBe(
        trimmedInstructions !== '' ? trimmedInstructions : HANDOFF_CUE
      );
    }
  );

  it('does not add a cue when stripping a transfer-only turn exposes the user request', async () => {
    const { requests } = await executeHandoff({ preamble: '' });
    expect(requests).toHaveLength(2);
    expect(requests[1].at(-1)?.getType()).toBe('human');
    expect(requests[1].at(-1)?.content).toBe(USER_REQUEST);
    expect(requests[1].some((message) => message.content === HANDOFF_CUE)).toBe(
      false
    );
  });

  it.each([undefined, 'Analyze the lookup result.'])(
    'preserves a retained tool-result tail and valid role ordering with instructions %j',
    async (instructions) => {
      const lookup = new DynamicStructuredTool({
        name: 'lookup',
        description: 'Look up organization usage',
        schema: { type: 'object', properties: {}, required: [] },
        func: async (): Promise<string> => 'Organization usage: 42',
      });
      const { requests } = await executeHandoff({
        agents: [
          createAgent('router'),
          { ...createAgent('recipient'), tools: [lookup] },
        ],
        preamble: '',
        messages: [
          new HumanMessage(USER_REQUEST),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'lookup-call', name: 'lookup', args: {} }],
          }),
          new ToolMessage({
            content: 'Organization usage: 42',
            tool_call_id: 'lookup-call',
            name: 'lookup',
          }),
        ],
        edges: [
          {
            from: 'router',
            to: 'recipient',
            edgeType: 'handoff',
            prompt: 'Instructions for the receiving agent',
          },
        ],
        toolCalls: [
          {
            id: 'transfer',
            name: `${Constants.LC_TRANSFER_TO_}recipient`,
            args: instructions === undefined ? {} : { instructions },
          },
        ],
      });
      expect(requests).toHaveLength(2);
      const received = requests[1];
      expect(getBufferString(received)).toContain('Organization usage: 42');
      expect(received.some((message) => message.content === HANDOFF_CUE)).toBe(
        false
      );
      if (instructions === undefined) {
        expect(received.at(-1)?.getType()).toBe('tool');
      } else {
        expect(received.slice(-3).map((message) => message.getType())).toEqual([
          'tool',
          'ai',
          'human',
        ]);
        expect(received.at(-1)?.content).toBe(instructions);
      }
    }
  );

  it('grounds an instructionless conditional handoff', async () => {
    const { requests } = await executeHandoff({
      edges: [
        { from: 'router', to: 'recipient', condition: () => 'recipient' },
      ],
      toolCalls: [{ id: 'transfer', name: 'conditional_transfer', args: {} }],
    });
    expect(requests).toHaveLength(2);
    expect(requests[1].at(-1)?.content).toBe(HANDOFF_CUE);
  });

  it('reconstructs the cue from legacy handoff history without run-produced message IDs', async () => {
    const transfer = {
      id: 'persisted-transfer',
      name: `${Constants.LC_TRANSFER_TO_}router`,
      args: {},
    };
    const messages = [
      new HumanMessage({ content: USER_REQUEST, id: 'persisted-user' }),
      new AIMessage({
        content: PREAMBLE,
        tool_calls: [transfer],
        id: 'persisted-assistant',
      }),
      new ToolMessage({
        content: 'Successfully transferred to router',
        name: transfer.name,
        tool_call_id: transfer.id,
        id: 'persisted-result',
      }),
    ];
    const originalMessages = messages.map((message) => message.toDict());
    const { requests } = await executeHandoff({
      messages,
      toolCalls: [],
      edges: [
        { from: 'router', to: 'recipient', edgeType: 'handoff' },
        { from: 'recipient', to: 'router', edgeType: 'handoff' },
      ],
    });
    expect(requests).toHaveLength(1);
    expect(requests[0].at(-1)?.content).toBe(HANDOFF_CUE);
    expect(messages.map((message) => message.toDict())).toEqual(
      originalMessages
    );
  });

  it('does not inject a cue when a conditional transfer is declined', async () => {
    const { requests } = await executeHandoff({
      edges: [{ from: 'router', to: 'recipient', condition: () => false }],
      toolCalls: [{ id: 'transfer', name: 'conditional_transfer', args: {} }],
    });
    expect(requests).toHaveLength(2);
    expect(requests[1].at(-1)?.getType()).toBe('tool');
    expect(requests[1].some((message) => message.content === HANDOFF_CUE)).toBe(
      false
    );
  });

  it('gives each parallel recipient one independently created cue', async () => {
    const { requests } = await executeHandoff({
      agents: [
        createAgent('router'),
        createAgent('left'),
        createAgent('right'),
      ],
      edges: [
        { from: 'router', to: 'left', edgeType: 'handoff' },
        { from: 'router', to: 'right', edgeType: 'handoff' },
      ],
      toolCalls: [
        {
          id: 'left-transfer',
          name: `${Constants.LC_TRANSFER_TO_}left`,
          args: {},
        },
        {
          id: 'right-transfer',
          name: `${Constants.LC_TRANSFER_TO_}right`,
          args: {},
        },
      ],
    });
    expect(requests).toHaveLength(3);
    for (const received of requests.slice(1)) {
      expect(received.at(-1)?.content).toBe(HANDOFF_CUE);
      expect(
        received.filter((message) => message.content === HANDOFF_CUE)
      ).toHaveLength(1);
    }
    expect(requests[1].at(-1)).not.toBe(requests[2].at(-1));
  });
});

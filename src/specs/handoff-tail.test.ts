import { join } from 'node:path';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { Command, MemorySaver } from '@langchain/langgraph';
import { DynamicStructuredTool } from '@langchain/core/tools';
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  getBufferString,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { inspectProviderMessageProvenance } from '@/messages/provenance';
import { Constants, Providers } from '@/common';
import { createAgentSession } from '@/session';
import * as providers from '@/llm/providers';
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
  overrideModel,
}: {
  agents?: t.AgentInputs[];
  edges?: t.GraphEdge[];
  toolCalls?: ToolCall[];
  preamble?: string;
  messages?: BaseMessage[];
  overrideModel?: FakeChatModel;
} = {}): Promise<{
  requests: BaseMessage[][];
  run: Run<t.BaseGraphState>;
  countedMessages: BaseMessage[];
}> {
  const countedMessages: BaseMessage[] = [];
  const checkpointer = new MemorySaver();
  const run = await Run.create({
    runId: `handoff-tail-${Math.random()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
      compileOptions: { checkpointer },
    },
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
  const model =
    overrideModel ??
    new FakeChatModel({
      responses: [preamble, 'Analysis complete', 'Second analysis complete'],
      toolCalls,
    });
  const streamSpy = jest.spyOn(model, '_streamResponseChunks');
  run.Graph.overrideModel = model;
  await run.processStream(
    { messages },
    {
      configurable: { thread_id: 'handoff-tail-thread' },
      version: 'v2',
      durability: 'sync',
    }
  );
  expect(getBufferString(run.getRunMessages() ?? [])).not.toContain(
    HANDOFF_CUE
  );
  let checkpointCount = 0;
  for await (const tuple of checkpointer.list({
    configurable: { thread_id: 'handoff-tail-thread' },
  })) {
    checkpointCount++;
    expect(JSON.stringify(tuple.checkpoint.channel_values)).not.toContain(
      HANDOFF_CUE
    );
    expect(JSON.stringify(tuple.pendingWrites)).not.toContain(HANDOFF_CUE);
  }
  expect(checkpointCount).toBeGreaterThan(0);
  return {
    requests: streamSpy.mock.calls.map(([messages]) => messages),
    run,
    countedMessages,
  };
}

describe('Handoff message tails', () => {
  it.each([false, true])(
    'does not persist or replay transport cues in JSONL sessions (parallel=%s)',
    async (parallel) => {
      const requests: BaseMessage[][] = [];
      const destinations = parallel ? ['left', 'right'] : ['recipient'];
      class SessionModel extends FakeChatModel {
        constructor() {
          super({ responses: ['Analysis complete'] });
        }

        override bindTools(): this {
          return this;
        }

        override async *_streamResponseChunks(
          messages: BaseMessage[],
          options: this['ParsedCallOptions'],
          runManager?: CallbackManagerForLLMRun
        ): AsyncGenerator<ChatGenerationChunk> {
          requests.push(messages);
          const transferring = messages.at(-1)?.content === USER_REQUEST;
          const scripted = new FakeChatModel({
            responses: [transferring ? PREAMBLE : 'Analysis complete'],
            toolCalls: transferring
              ? destinations.map((destination) => ({
                id: `transfer-${destination}`,
                name: `${Constants.LC_TRANSFER_TO_}${destination}`,
                args: {},
              }))
              : [],
          });
          yield* scripted._streamResponseChunks(messages, options, runManager);
        }
      }
      const modelSpy = jest
        .spyOn(providers, 'getChatModelClass')
        .mockReturnValue(SessionModel as never);
      const dir = await mkdtemp(join(process.cwd(), '.handoff-session-'));
      try {
        const config = {
          cwd: dir,
          sessionPath: join(dir, 'history.jsonl'),
          checkpointing: false as const,
          graphConfig: {
            type: 'multi-agent' as const,
            agents: [
              createAgent('router'),
              ...destinations.map((id) => createAgent(id)),
            ],
            edges: destinations.map(
              (to): t.GraphEdge => ({ from: 'router', to, edgeType: 'handoff' })
            ),
          },
        };
        const session = await createAgentSession(config);
        const result = await session.run(USER_REQUEST);
        expect(requests).toHaveLength(destinations.length + 1);
        for (const request of requests.slice(1)) {
          expect(request.at(-1)?.content).toBe(HANDOFF_CUE);
        }
        expect(getBufferString(result.messages)).not.toContain(HANDOFF_CUE);
        expect(await readFile(config.sessionPath, 'utf8')).not.toContain(
          HANDOFF_CUE
        );

        const reopened = await createAgentSession(config);
        const count = requests.length;
        await reopened.run('What should we do next?');
        expect(requests).toHaveLength(count + 1);
        expect(getBufferString(requests.at(-1)!)).toContain(
          'Analysis complete'
        );
        expect(getBufferString(requests.at(-1)!)).not.toContain(HANDOFF_CUE);
        expect(await readFile(config.sessionPath, 'utf8')).not.toContain(
          HANDOFF_CUE
        );
      } finally {
        modelSpy.mockRestore();
        await rm(dir, { recursive: true, force: true });
      }
    }
  );

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

  it('reconstructs the wire cue when a fresh Run resumes a checkpointed handoff', async () => {
    const checkpointer = new MemorySaver();
    const graphConfig: t.RunConfig['graphConfig'] = {
      type: 'multi-agent',
      agents: [createAgent('router'), createAgent('recipient')],
      edges: [{ from: 'router', to: 'recipient', edgeType: 'handoff' }],
      compileOptions: { checkpointer, interruptBefore: ['recipient'] },
    };
    const config = {
      configurable: { thread_id: 'paused-handoff' },
      version: 'v2' as const,
      durability: 'sync' as const,
    };
    const first = await Run.create({
      runId: 'paused-handoff',
      graphConfig,
      skipCleanup: true,
    });
    first.Graph?.overrideTestModel([PREAMBLE], 0, [
      {
        id: 'paused-transfer',
        name: `${Constants.LC_TRANSFER_TO_}recipient`,
        args: {},
      },
    ]);
    await first.processStream(
      { messages: [new HumanMessage(USER_REQUEST)] },
      config
    );
    const paused = await checkpointer.getTuple(config);
    expect(JSON.stringify(paused?.checkpoint.channel_values)).toContain(
      'paused-transfer'
    );
    expect(JSON.stringify(paused?.checkpoint.channel_values)).not.toContain(
      HANDOFF_CUE
    );

    const resumed = await Run.create({
      runId: 'resumed-handoff',
      skipCleanup: true,
      graphConfig: { ...graphConfig, compileOptions: { checkpointer } },
    });
    if (resumed.Graph == null) {
      throw new Error('Expected graph');
    }
    const model = new FakeChatModel({ responses: ['Analysis complete'] });
    const spy = jest.spyOn(model, '_streamResponseChunks');
    resumed.Graph.overrideModel = model;
    await resumed.processStream(new Command({ resume: true }), config);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0].at(-1)?.content).toBe(HANDOFF_CUE);
    expect(getBufferString(resumed.getRunMessages() ?? [])).not.toContain(
      HANDOFF_CUE
    );
    const completed = await checkpointer.getTuple(config);
    expect(JSON.stringify(completed?.checkpoint.channel_values)).toContain(
      'Analysis complete'
    );
    expect(JSON.stringify(completed?.checkpoint.channel_values)).not.toContain(
      HANDOFF_CUE
    );
  });

  it('does not carry the handoff cue into a later direct re-entry', async () => {
    const { requests } = await executeHandoff({
      edges: [
        { from: 'router', to: 'recipient', edgeType: 'handoff' },
        { from: 'recipient', to: 'router', edgeType: 'direct' },
      ],
    });
    expect(requests).toHaveLength(3);
    expect(requests[1].at(-1)?.content).toBe(HANDOFF_CUE);
    expect(requests[2].at(-1)?.getType()).toBe('ai');
    expect(getBufferString(requests[2])).not.toContain(HANDOFF_CUE);
  });

  it('keeps cues transient through another handoff and a recipient tool iteration', async () => {
    const lookup = new DynamicStructuredTool({
      name: 'lookup',
      description: 'Look up usage',
      schema: { type: 'object', properties: {}, required: [] },
      func: async (): Promise<string> => 'Usage: 42',
    });
    class ChainedModel extends FakeChatModel {
      private turn = 0;
      override async *_streamResponseChunks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
      ): AsyncGenerator<ChatGenerationChunk> {
        const scripts = [
          { text: PREAMBLE, name: `${Constants.LC_TRANSFER_TO_}middle` },
          {
            text: 'Delegating the analysis',
            name: `${Constants.LC_TRANSFER_TO_}recipient`,
          },
          { text: 'Looking up usage', name: 'lookup' },
          { text: 'Analysis complete', name: undefined },
        ];
        const script = scripts.at(this.turn++);
        if (script == null) {
          throw new Error('Unexpected extra model turn');
        }
        const scripted = new FakeChatModel({
          responses: [script.text],
          toolCalls:
            script.name == null
              ? []
              : [{ id: `call-${this.turn}`, name: script.name, args: {} }],
        });
        yield* scripted._streamResponseChunks(messages, options, runManager);
      }
    }
    const { requests } = await executeHandoff({
      agents: [
        createAgent('router'),
        createAgent('middle'),
        { ...createAgent('recipient'), tools: [lookup] },
      ],
      edges: [
        { from: 'router', to: 'middle', edgeType: 'handoff' },
        { from: 'middle', to: 'recipient', edgeType: 'handoff' },
      ],
      overrideModel: new ChainedModel({ responses: [] }),
    });
    expect(requests).toHaveLength(4);
    for (const request of requests.slice(1, 3)) {
      expect(request.at(-1)?.content).toBe(HANDOFF_CUE);
      expect(
        request.filter((message) => message.content === HANDOFF_CUE)
      ).toHaveLength(1);
    }
    expect(requests[3].at(-1)?.getType()).toBe('tool');
    expect(getBufferString(requests[3])).not.toContain(HANDOFF_CUE);
  });

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

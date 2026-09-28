import { CallbackHandler } from '@langfuse/langchain';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type * as t from '@/types';
import { LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT } from '@/langfuseToolOutputTracing';
import { Providers } from '@/common';
import { Run } from '@/run';

const mockModelInputs: string[] = [];
const MODEL_REPLY = 'Counted EU orders from the private table';

jest.mock('@/llm/init', () => ({
  initializeModel: jest.fn(() => {
    const { FakeListChatModel } = jest.requireActual(
      '@langchain/core/utils/testing'
    );
    const model = new FakeListChatModel({
      responses: ['Counted EU orders from the private table'],
    });
    const invoke = model.invoke.bind(model);
    model.invoke = (messages: BaseMessage[], config: unknown) => {
      mockModelInputs.push(
        messages.map((message) => String(message.content)).join('\n')
      );
      return invoke(messages, config);
    };
    return model;
  }),
}));

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation(() => ({
    forceFlush: jest.fn(),
    onEnd: jest.fn(),
    onStart: jest.fn(),
    shutdown: jest.fn(),
  })),
  isDefaultExportSpan: jest.fn(() => false),
}));

/** What the Langfuse handler would export for each callback. */
const traced = {
  chatInputs: [] as string[],
  llmOutputs: [] as string[],
  chainOutputs: [] as string[],
};

const SECRET_ROWS = 'PRIVATE_ROWS_42113';
const REDACTING_LANGFUSE: t.LangfuseConfig = {
  publicKey: 'pk-trace-mask',
  secretKey: 'sk-trace-mask',
  baseUrl: 'https://langfuse.test',
  toolOutputTracing: {
    redactedToolNames: ['run_select_query'],
    redactedToolNameMatchMode: 'partial',
  },
};
const CHAIN_OPTIONS = {
  configurable: { thread_id: 'thread-1', user_id: 'user-1' },
};

async function createRun(
  langfuse: t.LangfuseConfig = REDACTING_LANGFUSE
): Promise<Run<never>> {
  const run = await Run.create({
    runId: 'trace-mask-run',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent-1',
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4.1-mini' },
          tools: [],
        },
      ],
    },
    langfuse,
  });
  if (run.Graph != null) {
    run.Graph.messages = [new HumanMessage('How many EU orders came in?')];
  }
  return run;
}

describe('label calls under a tool-output redaction policy', () => {
  beforeEach(() => {
    mockModelInputs.length = 0;
    traced.chatInputs.length = 0;
    traced.llmOutputs.length = 0;
    traced.chainOutputs.length = 0;
    jest
      .spyOn(CallbackHandler.prototype, 'handleChatModelStart')
      .mockImplementation(async (_llm, messages) => {
        traced.chatInputs.push(
          messages
            .flat()
            .map((message) => String(message.content))
            .join('\n')
        );
      });
    jest
      .spyOn(CallbackHandler.prototype, 'handleLLMEnd')
      .mockImplementation(async (output: LLMResult) => {
        traced.llmOutputs.push(
          output.generations
            .flat()
            .map((generation) => generation.text)
            .join('')
        );
      });
    jest
      .spyOn(CallbackHandler.prototype, 'handleChainEnd')
      .mockImplementation(async (outputs) => {
        traced.chainOutputs.push(JSON.stringify(outputs));
      });
    jest
      .spyOn(CallbackHandler.prototype, 'handleChainStart')
      .mockImplementation(async () => undefined);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('sends redacted tool output to the label model but not to the trace', async () => {
    const run = await createRun();

    const result = await run.generateActivityLabel({
      provider: Providers.OPENAI,
      entries: [
        {
          toolName: 'run_select_query',
          toolInput: { query: 'SELECT count() FROM orders' },
          toolOutput: SECRET_ROWS,
          status: 'success',
        },
      ],
      thinkingExcerpts: ['Checking the EU order count'],
      chainOptions: CHAIN_OPTIONS,
    });

    expect(result).toEqual({ label: MODEL_REPLY });
    expect(mockModelInputs[0]).toContain(SECRET_ROWS);
    expect(mockModelInputs[0]).toContain('Checking the EU order count');
    expect(traced.chatInputs).toHaveLength(1);
    expect(traced.chatInputs[0]).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(traced.chatInputs[0]).not.toContain(SECRET_ROWS);
    expect(traced.chatInputs[0]).not.toContain('Checking the EU order count');
    expect(traced.llmOutputs).toEqual([LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT]);
    expect(JSON.stringify(traced)).not.toContain(MODEL_REPLY);
  });

  it('traces the label call unchanged when no evidence is withheld', async () => {
    const run = await createRun();

    await run.generateActivityLabel({
      provider: Providers.OPENAI,
      entries: [
        {
          toolName: 'web_search',
          toolInput: { q: 'eu order holidays' },
          toolOutput: 'public calendar',
          status: 'success',
        },
      ],
      chainOptions: CHAIN_OPTIONS,
    });

    expect(traced.chatInputs[0]).toContain('public calendar');
    expect(traced.llmOutputs).toEqual([MODEL_REPLY]);
  });

  it('titles a labels-only phase from its child labels while the trace sees neither', async () => {
    const run = await createRun();

    const result = await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        { agentId: 'agent-1', label: `Found ${SECRET_ROWS} in orders` },
        { agentId: 'agent-1', label: 'Grouped orders by region' },
      ],
      assistantContext: ['Comparing regions next'],
      chainOptions: CHAIN_OPTIONS,
    });

    expect(result).toEqual({ label: MODEL_REPLY });
    expect(mockModelInputs[0]).toContain(`Found ${SECRET_ROWS} in orders`);
    expect(traced.chainOutputs.length).toBeGreaterThan(0);
    expect(mockModelInputs[0]).toContain('Comparing regions next');
    expect(traced.chatInputs[0]).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(JSON.stringify(traced)).not.toContain(SECRET_ROWS);
    expect(JSON.stringify(traced)).not.toContain('Grouped orders by region');
    expect(JSON.stringify(traced)).not.toContain(MODEL_REPLY);
  });

  it('labels reasoning for the model while the trace sees only the redaction text', async () => {
    const run = await createRun();

    const result = await run.generateReasoningLabel({
      provider: Providers.OPENAI,
      visibleReasoning: `The query returned ${SECRET_ROWS}, so EU leads`,
      reasoningStepId: 'reasoning-step-1',
      revision: 0,
      chainOptions: CHAIN_OPTIONS,
    });

    expect(result.label).toBe(MODEL_REPLY);
    expect(mockModelInputs[0]).toContain(SECRET_ROWS);
    expect(traced.chatInputs).toHaveLength(1);
    expect(traced.llmOutputs).toEqual([LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT]);
    expect(JSON.stringify(traced)).not.toContain(SECRET_ROWS);
    expect(JSON.stringify(traced)).not.toContain(MODEL_REPLY);
  });

  it('applies the union of agent policies to the trace of an unattributed phase', async () => {
    const run = await Run.create({
      runId: 'trace-mask-union-run',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent-1',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
          },
          {
            agentId: 'agent-2',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
            langfuse: {
              toolOutputTracing: { redactedToolNames: ['secret_tool'] },
            },
          },
        ],
        edges: [],
      },
      langfuse: {
        publicKey: 'pk-trace-mask',
        secretKey: 'sk-trace-mask',
        baseUrl: 'https://langfuse.test',
      },
    });
    if (run.Graph != null) {
      run.Graph.messages = [new HumanMessage('Check both agents')];
    }

    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        {
          agentId: 'agent-1',
          entries: [
            {
              toolName: 'public_lookup',
              toolInput: { id: 'one' },
              toolOutput: 'public-one',
              status: 'success',
            },
          ],
        },
        {
          entries: [
            {
              toolName: 'secret_tool',
              toolInput: { key: 'k' },
              toolOutput: 'STRICT_AGENT_SECRET',
              status: 'success',
            },
          ],
        },
      ],
      chainOptions: CHAIN_OPTIONS,
    });

    expect(mockModelInputs[0]).toContain('STRICT_AGENT_SECRET');
    expect(traced.chatInputs[0]).toContain('public-one');
    expect(traced.chatInputs[0]).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(JSON.stringify(traced)).not.toContain('STRICT_AGENT_SECRET');
  });

  it('masks assistant context in the trace when omitted activities have no agent list', async () => {
    const run = await createRun();

    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        {
          agentId: 'agent-1',
          entries: [
            {
              toolName: 'public_lookup',
              toolInput: { id: 'one' },
              toolOutput: 'public-one',
              status: 'success',
            },
          ],
        },
        {
          agentId: 'agent-1',
          entries: [
            {
              toolName: 'public_lookup',
              toolInput: { id: 'two' },
              toolOutput: 'public-two',
              status: 'success',
            },
          ],
        },
      ],
      totalActivityCount: 3,
      assistantContext: ['OMITTED_AGENT_SECRET'],
      chainOptions: CHAIN_OPTIONS,
    });

    expect(mockModelInputs[0]).toContain('OMITTED_AGENT_SECRET');
    expect(traced.chatInputs[0]).toContain('public-one');
    expect(JSON.stringify(traced)).not.toContain('OMITTED_AGENT_SECRET');
  });
});

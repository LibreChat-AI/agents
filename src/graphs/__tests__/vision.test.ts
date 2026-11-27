import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  AIMessageChunk,
  getBufferString,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { OMITTED_IMAGE_TEXT } from '@/messages/vision';
import { MultiAgentGraph } from '../MultiAgentGraph';
import { Constants, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import * as init from '@/llm/init';
import { Run } from '@/run';

const image = {
  type: 'image_url',
  image_url: { url: 'data:image/png;base64,AA==' },
};
const question = { type: 'text', text: 'What does this chart show?' };
const marker = { type: 'text', text: OMITTED_IMAGE_TEXT };

const agent = (
  agentId: string,
  overrides: Pick<t.AgentInputs, 'vision' | 'useLegacyContent'> = {}
): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
  ...overrides,
});

/** The edge shape LibreChat builds for an agent chain. */
const chainEdge: t.GraphEdge = {
  from: 'A',
  to: 'B',
  edgeType: 'direct',
  prompt: (messages: BaseMessage[], startIndex: number): string =>
    `Previous context:\n${getBufferString(messages.slice(startIndex))}`,
  excludeResults: true,
};

const summaryStreamConfig = {
  configurable: { thread_id: 'text-only-summary' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

const imageTurn = (): HumanMessage =>
  new HumanMessage({ content: [question, image] });

const sentContent = (messages: BaseMessage[]): BaseMessage['content'][] =>
  messages.map((message) => message.content);

const hasImage = (messages: BaseMessage[]): boolean =>
  messages.some(
    (message) =>
      Array.isArray(message.content) &&
      message.content.some((block) => block.type === 'image_url')
  );

class CapturingChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  constructor(responses: string[]) {
    super({ responses });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

/** Transfers to B on its first call, then answers as B. */
class HandoffChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  constructor() {
    super({ responses: ['unused'] });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    if (this.invocations.length === 1) {
      yield this._createResponseChunk('', [
        {
          id: 'transfer_1',
          name: `${Constants.LC_TRANSFER_TO_}B`,
          args: JSON.stringify({ instructions: 'Summarize the chart.' }),
          index: 0,
          type: 'tool_call_chunk',
        },
      ]);
      void runManager?.handleLLMNewToken('');
      return;
    }
    yield this._createResponseChunk('B answered');
    void runManager?.handleLLMNewToken('B answered');
  }
}

/** Answers every call with the same text and remembers what it was sent. */
class RecordingModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly calls: BaseMessage[][] = [];

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.calls.push(messages);
    return new AIMessageChunk({ content: 'done' });
  }
}

/** Runs one summarize-only turn for a text-only agent and returns what the summarizer was sent. */
async function captureSummaryRequest(
  overrides: Pick<t.AgentInputs, 'useLegacyContent'>
): Promise<BaseMessage[]> {
  const run = await Run.create<t.IState>({
    runId: 'text-only-summary',
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.ANTHROPIC,
        disableStreaming: true,
        streamUsage: false,
      },
      maxContextTokens: 100_000,
      summarizationEnabled: true,
      summarizeOnly: true,
      vision: false,
      ...overrides,
    },
    tokenCounter: () => 10,
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  run.Graph.overrideModel = new RecordingModel();
  const summarizer = new RecordingModel();
  const spy = jest.spyOn(init, 'initializeModel').mockReturnValue(summarizer);
  try {
    await run.processStream(
      {
        messages: [
          imageTurn(),
          new AIMessage('It shows revenue by quarter.'),
          new HumanMessage('Which quarter was best?'),
          new AIMessage('The fourth.'),
          new HumanMessage('Thanks.'),
          new AIMessage('You are welcome.'),
        ],
      },
      summaryStreamConfig
    );
  } finally {
    spy.mockRestore();
  }
  expect(summarizer.calls).toHaveLength(1);
  return summarizer.calls[0];
}

describe('an agent whose model cannot read images', () => {
  it('@scenario:text-only-chain gets a marker while the vision agent before it gets the image', async () => {
    const model = new CapturingChatModel(['A read the chart', 'B answered']);
    const graph = new MultiAgentGraph({
      runId: 'text-only-chain',
      agents: [agent('A'), agent('B', { vision: false })],
      edges: [chainEdge],
    });
    graph.overrideModel = model;
    const turn = imageTurn();

    await graph
      .createWorkflow()
      .invoke(
        { messages: [turn] },
        { configurable: { thread_id: 'text-only-chain' } }
      );

    expect(model.invocations).toHaveLength(2);
    const [visionCall, textOnlyCall] = model.invocations;
    expect(hasImage(visionCall)).toBe(true);
    expect(hasImage(textOnlyCall)).toBe(false);
    expect(sentContent(textOnlyCall)).toContainEqual([question, marker]);
    expect(turn.content).toEqual([question, image]);
  });

  it('@scenario:text-only-handoff gets a marker when a vision agent hands off to it', async () => {
    const model = new HandoffChatModel();
    const graph = new MultiAgentGraph({
      runId: 'text-only-handoff',
      agents: [agent('A'), agent('B', { vision: false })],
      edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
    });
    graph.overrideModel = model;

    await graph
      .createWorkflow()
      .invoke(
        { messages: [imageTurn()] },
        { configurable: { thread_id: 'text-only-handoff' } }
      );

    expect(model.invocations).toHaveLength(2);
    const [routingCall, receivingCall] = model.invocations;
    expect(hasImage(routingCall)).toBe(true);
    expect(hasImage(receivingCall)).toBe(false);
    expect(sentContent(receivingCall)).toContainEqual([question, marker]);
  });

  it('gets the turn as one string when it formats content as strings', async () => {
    const model = new CapturingChatModel(['A read the chart', 'B answered']);
    const graph = new MultiAgentGraph({
      runId: 'text-only-legacy-chain',
      agents: [
        agent('A'),
        agent('B', { vision: false, useLegacyContent: true }),
      ],
      edges: [chainEdge],
    });
    graph.overrideModel = model;

    await graph
      .createWorkflow()
      .invoke(
        { messages: [imageTurn()] },
        { configurable: { thread_id: 'text-only-legacy-chain' } }
      );

    expect(sentContent(model.invocations[1])).toContain(
      `${question.text}\n${OMITTED_IMAGE_TEXT}`
    );
  });

  it('keeps an image turn whose marker fits instead of pruning it at image cost', async () => {
    /** Prices an image the way a host count does; anything else is cheap. */
    const tokenCounter: t.TokenCounter = (message) =>
      hasImage([message]) ? 50_000 : 10;
    const run = await Run.create<t.IState>({
      runId: 'text-only-prune',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000,
        vision: false,
      },
      tokenCounter,
      indexTokenCountMap: { 0: 50_000, 1: 10, 2: 10 },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new RecordingModel();
    run.Graph.overrideModel = model;

    await run.processStream(
      {
        messages: [
          imageTurn(),
          new AIMessage('It shows revenue by quarter.'),
          new HumanMessage('Which quarter was best?'),
        ],
      },
      { ...summaryStreamConfig, configurable: { thread_id: 'text-only-prune' } }
    );

    expect(model.calls).toHaveLength(1);
    expect(sentContent(model.calls[0])).toContainEqual([question, marker]);
  });

  it('keeps the history when fading trims a tool result that holds an image', async () => {
    /** Prices an image the way a host count does; anything else is cheap. */
    const tokenCounter: t.TokenCounter = (message) =>
      hasImage([message]) ? 50_000 : 10;
    const run = await Run.create<t.IState>({
      runId: 'text-only-fading',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000,
        vision: false,
      },
      tokenCounter,
      indexTokenCountMap: { 0: 10, 1: 10, 2: 50_000, 3: 10, 4: 10 },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new RecordingModel();
    run.Graph.overrideModel = model;

    await run.processStream(
      {
        messages: [
          new HumanMessage('Chart the revenue.'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'call_1', name: 'chart', args: {} }],
          }),
          new ToolMessage({
            tool_call_id: 'call_1',
            name: 'chart',
            content: [{ type: 'text', text: 'Q'.repeat(20_000) }, image],
          }),
          new AIMessage('It shows revenue by quarter.'),
          new HumanMessage('Which quarter was best?'),
        ],
      },
      {
        ...summaryStreamConfig,
        configurable: { thread_id: 'text-only-fading' },
      }
    );

    expect(model.calls).toHaveLength(1);
    const sent = JSON.stringify(model.calls[0]);
    expect(sent).toContain('[truncated: ');
    expect(sent).toContain('Chart the revenue.');
  });

  it('gets a marker for a folded tool image when it has no tools and no context limit', async () => {
    const data = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk';
    const run = await Run.create<t.IState>({
      runId: 'text-only-fold',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          disableStreaming: true,
          streamUsage: false,
        },
        vision: false,
      },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new RecordingModel();
    run.Graph.overrideModel = model;

    await run.processStream(
      {
        messages: [
          new HumanMessage('Chart the revenue.'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'call_1', name: 'chart', args: {} }],
          }),
          new ToolMessage({
            tool_call_id: 'call_1',
            name: 'chart',
            content: [
              { type: 'text', text: 'Here is the chart.' },
              { type: 'media', mimeType: 'image/png', data },
            ],
          }),
          new HumanMessage('What does it show?'),
        ],
      },
      { ...summaryStreamConfig, configurable: { thread_id: 'text-only-fold' } }
    );

    expect(model.calls).toHaveLength(1);
    const sent = JSON.stringify(model.calls[0]);
    expect(sent).not.toContain(data);
    expect(sent).toContain(OMITTED_IMAGE_TEXT);
  });

  it('keeps the text of a tool artifact whose image would not fit at image cost', async () => {
    /** Prices an image the way a host count does; anything else is cheap. */
    const tokenCounter: t.TokenCounter = (message) =>
      hasImage([message]) ? 50_000 : 10;
    const run = await Run.create<t.IState>({
      runId: 'text-only-artifact',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000,
        vision: false,
        tools: [
          tool(async () => 'unused', {
            name: 'screenshot',
            description: 'Takes a screenshot of the page',
            schema: z.object({}),
          }),
        ],
      },
      tokenCounter,
      indexTokenCountMap: { 0: 10, 1: 10, 2: 10 },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new RecordingModel();
    run.Graph.overrideModel = model;
    const toolMessage = new ToolMessage({
      tool_call_id: 'call_1',
      name: 'screenshot',
      content: 'Screenshot taken.',
      artifact: {
        content: [
          image,
          { type: 'text', text: 'The page shows a login form.' },
        ],
      },
    });

    await run.processStream(
      {
        messages: [
          new HumanMessage('Open the page.'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'call_1', name: 'screenshot', args: {} }],
          }),
          toolMessage,
        ],
      },
      {
        ...summaryStreamConfig,
        configurable: { thread_id: 'text-only-artifact' },
      }
    );

    expect(model.calls).toHaveLength(1);
    const sent = JSON.stringify(model.calls[0]);
    expect(sent).toContain('The page shows a login form.');
    expect(sent).toContain(OMITTED_IMAGE_TEXT);
    expect(toolMessage.artifact.content[0]).toBe(image);
  });

  it('@scenario:text-only-summary gets a marker in the history its summary request sends', async () => {
    const [firstTurn] = await captureSummaryRequest({});

    expect(firstTurn.content).toEqual([question, marker]);
  });

  it('gets that history turn as one string when it formats content as strings', async () => {
    const [firstTurn] = await captureSummaryRequest({ useLegacyContent: true });

    expect(firstTurn.content).toBe(`${question.text}\n${OMITTED_IMAGE_TEXT}`);
  });
});

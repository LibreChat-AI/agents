import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import {
  AIMessage,
  HumanMessage,
  BaseMessage,
} from '@langchain/core/messages';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';
import { RequiredHostEventError } from '@/utils';

class CapturingFakeChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

class RawDispatchFakeChatModel extends CapturingFakeChatModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    await dispatchCustomEvent(
      GraphEvents.ON_AGENT_LOG,
      { type: 'raw_custom_event' },
      options as RunnableConfig
    );
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

const charCounter: t.TokenCounter = (msg: BaseMessage): number => {
  const content = msg.content;
  if (typeof content === 'string') {
    return content.length + 3;
  }
  return 3;
};

const llmConfig: t.LLMConfig = {
  provider: Providers.OPENAI,
  streaming: true,
  streamUsage: false,
};

const streamConfig = {
  configurable: { thread_id: 'context-usage-event' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

describe('ON_CONTEXT_USAGE event', () => {
  jest.setTimeout(15000);

  it('dispatches a post-prune context snapshot per model call', async () => {
    const received: t.ContextUsageEvent[] = [];
    const maxContextTokens = 4000;

    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-event',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
        maxContextTokens,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            received.push(data as unknown as t.ContextUsageEvent);
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });

    run.Graph?.overrideTestModel(['Hello there!'], 1);
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      streamConfig
    );

    expect(received).toHaveLength(1);
    const event = received[0];
    expect(event.runId).toBe('test-context-usage-event');
    expect(event.agentId).toBeDefined();
    expect(event.breakdown.maxContextTokens).toBe(maxContextTokens);
    expect(event.breakdown.instructionTokens).toBeGreaterThan(0);
    expect(event.breakdown.toolTokenCounts).toEqual({});
    expect(event.contextBudget).toBeGreaterThan(0);
    expect(event.contextBudget).toBeLessThanOrEqual(maxContextTokens);
    expect(event.effectiveInstructionTokens).toBeGreaterThan(0);
    expect(event.prePruneContextTokens).toBeGreaterThan(0);
    expect(event.remainingContextTokens).toBeGreaterThan(0);
    expect(event.remainingContextTokens).toBeLessThan(
      event.contextBudget as number
    );
    expect(event.breakdown.instructionTokens).toBe(
      event.effectiveInstructionTokens
    );
    expect(event.breakdown.availableForMessages).toBe(
      (event.contextBudget as number) -
        (event.effectiveInstructionTokens as number)
    );
    expect(event.breakdown.messageTokens).toBe(
      (event.contextBudget as number) -
        (event.effectiveInstructionTokens as number) -
        (event.remainingContextTokens as number)
    );
  });

  it('stops before the provider when a required context event fails', async () => {
    const cause = new Error('required context observer failed');
    const failure = new RequiredHostEventError({ cause });
    const model = new CapturingFakeChatModel({ responses: ['must not run'] });
    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-required-host-failure',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
        maxContextTokens: 4_000,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: async (): Promise<void> => {
            throw failure;
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });
    run.Graph!.overrideModel = model;

    await expect(
      run.processStream(
        { messages: [new HumanMessage('hello')] },
        streamConfig
      )
    ).rejects.toBe(failure);

    expect(failure.cause).toBe(cause);
    expect(model.invocations).toHaveLength(0);
  });

  it('keeps unmarked context observer failures best-effort', async () => {
    const failure = new Error('optional context observer failed');
    const model = new CapturingFakeChatModel({ responses: ['provider ran'] });
    const log = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-optional-host-failure',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
        maxContextTokens: 4_000,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: async (): Promise<void> => {
            throw failure;
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });
    run.Graph!.overrideModel = model;

    try {
      await run.processStream(
        { messages: [new HumanMessage('hello')] },
        streamConfig
      );
      expect(model.invocations).toHaveLength(1);
      expect(log).toHaveBeenCalledWith('Optional custom event handler failed.');
    } finally {
      log.mockRestore();
    }
  });

  it('keeps an unmarked observer failure best-effort on raw LangChain dispatch', async () => {
    const model = new RawDispatchFakeChatModel({ responses: ['provider ran'] });
    const warning = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const run = await Run.create<t.IState>({
      runId: 'test-raw-custom-event-optional-host-failure',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
        maxContextTokens: 4_000,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_AGENT_LOG]: {
          handle: async (): Promise<void> => {
            throw new Error('optional raw observer failure');
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });
    run.Graph!.overrideModel = model;

    try {
      await run.processStream(
        { messages: [new HumanMessage('hello')] },
        streamConfig
      );
      expect(model.invocations).toHaveLength(1);
      expect(warning).toHaveBeenCalledWith('Optional custom event handler failed.');
    } finally {
      warning.mockRestore();
    }
  });

  it('accounts for the retained exchange in the final provider-facing snapshot', async () => {
    const received: t.ContextUsageEvent[] = [];
    const retainedResult = `tool-result:${'x'.repeat(10_000)}`;
    const lookup = new DynamicStructuredTool({
      name: 'lookup',
      description: 'Returns a deterministic retained result.',
      schema: z.object({ query: z.string() }),
      func: async () => retainedResult,
    });
    const toolCall = {
      id: 'lookup-call',
      name: 'lookup',
      args: { query: 'retained context' },
      type: 'tool_call' as const,
    };
    const model = new CapturingFakeChatModel({
      responses: ['', 'done'],
      toolCalls: [toolCall],
    });

    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-event-tool-loop',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'Use lookup, then answer briefly.',
        maxContextTokens: 2_000,
        maxToolResultChars: 240,
        tools: [lookup],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: (_event, data): void => {
            if (data != null && 'breakdown' in data) {
              received.push(data);
            }
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });

    run.Graph!.overrideModel = model;
    await run.processStream(
      { messages: [new HumanMessage('Look this up.')] },
      {
        ...streamConfig,
        configurable: { thread_id: 'context-usage-event-tool-loop' },
      }
    );

    expect(model.invocations).toHaveLength(2);
    expect(received).toHaveLength(2);
    expect(received[0].breakdown.toolMessageTokens).toBe(0);
    expect(received[0].breakdown.toolMessageTokenCounts).toBeUndefined();

    const finalProviderMessages = model.invocations[1];
    const retainedToolMessages = finalProviderMessages.filter((message) => {
      if (message.getType() === 'tool') {
        return true;
      }
      if (message.getType() !== 'ai') {
        return false;
      }
      const aiMessage = message as AIMessage;
      return (
        (aiMessage.tool_calls?.length ?? 0) > 0 &&
        typeof aiMessage.content === 'string' &&
        !/\S/u.test(aiMessage.content)
      );
    });
    const providerToolTokens = retainedToolMessages.reduce(
      (total, message) => total + charCounter(message),
      0
    );
    const finalEvent = received[1];
    const retainedToolResult = retainedToolMessages.find(
      (message) => message.getType() === 'tool'
    );
    expect(retainedToolResult).toBeDefined();
    const resultTokenCount = charCounter(retainedToolResult!);
    expect(String(retainedToolResult?.content).length).toBeLessThan(
      retainedResult.length
    );
    expect(finalEvent.breakdown.toolMessageTokens).toBe(providerToolTokens);
    expect(resultTokenCount).toBeGreaterThan(0);
    expect(finalEvent.breakdown.toolMessageTokens).toBeGreaterThan(
      resultTokenCount
    );
    expect(finalEvent.breakdown.toolMessageTokenCounts).toEqual({
      lookup: resultTokenCount,
    });
  });

  it('does not dispatch when no tokenCounter is configured', async () => {
    const received: t.ContextUsageEvent[] = [];

    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-event-no-counter',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            received.push(data as unknown as t.ContextUsageEvent);
          },
        },
      },
    });

    run.Graph?.overrideTestModel(['Hello there!'], 1);
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      streamConfig
    );

    expect(received).toHaveLength(0);
  });
});

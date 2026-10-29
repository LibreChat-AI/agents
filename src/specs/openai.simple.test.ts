/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
// src/scripts/cli.test.ts
import { config } from 'dotenv';
config();
import { Calculator } from '@/tools/Calculator';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  ToolEndHandler,
  ModelEndHandler,
  createMetadataAggregator,
} from '@/events';
import { ContentTypes, GraphEvents, Providers, TitleMethod } from '@/common';
import { capitalizeFirstLetter, hasEnv } from './spec.utils';
import { createContentAggregator } from '@/stream';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

const provider = Providers.OPENAI;
const describeIfOpenAI = hasEnv('OPENAI_API_KEY') ? describe : describe.skip;

describeIfOpenAI(`${capitalizeFirstLetter(provider)} Streaming Tests`, () => {
  jest.setTimeout(30000);
  let run: Run<t.IState>;
  let runningHistory: BaseMessage[];
  let collectedUsage: UsageMetadata[];
  let conversationHistory: BaseMessage[];
  let aggregateContent: t.ContentAggregator;
  let contentParts: t.MessageContentComplex[];

  const config = {
    configurable: {
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  beforeEach(async () => {
    conversationHistory = [];
    collectedUsage = [];
    const { contentParts: cp, aggregateContent: ac } =
      createContentAggregator();
    contentParts = cp as t.MessageContentComplex[];
    aggregateContent = ac;
  });

  const onMessageDeltaSpy = jest.fn();
  const onRunStepSpy = jest.fn();

  afterAll(() => {
    onMessageDeltaSpy.mockReset();
    onRunStepSpy.mockReset();
  });

  const setupCustomHandlers = (): Record<
    string | GraphEvents,
    t.EventHandler
  > => ({
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        onRunStepSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        onMessageDeltaSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        _metadata?: Record<string, unknown>
      ): void => {
        // Handle tool start
      },
    },
  });

  test(`${capitalizeFirstLetter(provider)}: should process a simple message, generate title`, async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    const userMessage = 'hi';
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();
    const allTextParts = finalContentParts?.every(
      (part) => part.type === ContentTypes.TEXT
    );
    expect(allTextParts).toBe(true);
    expect(collectedUsage.length).toBeGreaterThan(0);
    expect(collectedUsage[0].input_tokens).toBeGreaterThan(0);
    expect(collectedUsage[0].output_tokens).toBeGreaterThan(0);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    conversationHistory.push(...(finalMessages ?? []));
    expect(conversationHistory.length).toBeGreaterThan(1);
    runningHistory = conversationHistory.slice();

    expect(onMessageDeltaSpy).toHaveBeenCalled();
    expect(onMessageDeltaSpy.mock.calls.length).toBeGreaterThan(1);
    expect(onMessageDeltaSpy.mock.calls[0][3]).toBeDefined(); // Graph exists

    expect(onRunStepSpy).toHaveBeenCalled();
    expect(onRunStepSpy.mock.calls.length).toBeGreaterThan(0);
    expect(onRunStepSpy.mock.calls[0][3]).toBeDefined(); // Graph exists

    const { handleLLMEnd, collected } = createMetadataAggregator();
    const titleResult = await run.generateTitle({
      provider,
      inputText: userMessage,
      titleMethod: TitleMethod.STRUCTURED,
      contentParts,
      clientOptions: llmConfig,
      chainOptions: {
        callbacks: [
          {
            handleLLMEnd,
          },
        ],
      },
    });

    expect(titleResult).toBeDefined();
    expect(titleResult.title).toBeDefined();
    expect(titleResult.language).toBeDefined();
    expect(collected).toBeDefined();
  });

  test(`${capitalizeFirstLetter(provider)}: should generate title using completion method`, async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id-completion',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    const userMessage = 'What is the weather like today?';
    conversationHistory = [];
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();

    const { handleLLMEnd, collected } = createMetadataAggregator();
    const titleResult = await run.generateTitle({
      provider,
      inputText: userMessage,
      titleMethod: TitleMethod.COMPLETION, // Using completion method
      contentParts,
      clientOptions: llmConfig,
      chainOptions: {
        callbacks: [
          {
            handleLLMEnd,
          },
        ],
      },
    });

    expect(titleResult).toBeDefined();
    expect(titleResult.title).toBeDefined();
    expect(titleResult.title).not.toBe('');
    // Completion method doesn't return language
    expect(titleResult.language).toBeUndefined();
    expect(collected).toBeDefined();
    console.log(`Completion method generated title: "${titleResult.title}"`);
  });

  test(`${capitalizeFirstLetter(provider)}: should follow-up`, async () => {
    console.log('Previous conversation length:', runningHistory.length);
    console.log(
      'Last message:',
      runningHistory[runningHistory.length - 1].content
    );
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    conversationHistory = runningHistory.slice();
    conversationHistory.push(new HumanMessage('how are you?'));

    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();
    const allTextParts = finalContentParts?.every(
      (part) => part.type === ContentTypes.TEXT
    );
    expect(allTextParts).toBe(true);
    expect(collectedUsage.length).toBeGreaterThan(0);
    expect(collectedUsage[0].input_tokens).toBeGreaterThan(0);
    expect(collectedUsage[0].output_tokens).toBeGreaterThan(0);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    expect(finalMessages?.length).toBeGreaterThan(0);
    console.log(
      `${capitalizeFirstLetter(provider)} follow-up message:`,
      finalMessages?.[finalMessages.length - 1]?.content
    );

    expect(onMessageDeltaSpy).toHaveBeenCalled();
    expect(onMessageDeltaSpy.mock.calls.length).toBeGreaterThan(1);

    expect(onRunStepSpy).toHaveBeenCalled();
    expect(onRunStepSpy.mock.calls.length).toBeGreaterThan(0);
  });

  test('should handle errors appropriately', async () => {
    // Test error scenarios
    await expect(async () => {
      await run.processStream(
        {
          messages: [],
        },
        {} as any
      );
    }).rejects.toThrow();
  });
});

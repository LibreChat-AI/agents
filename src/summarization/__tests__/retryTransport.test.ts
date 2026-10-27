import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { withSummarizationRetryOwner } from '@/summarization/retry';
import { createSummarizeNode } from '@/summarization/node';
import { AgentContext } from '@/agents/AgentContext';
import * as events from '@/utils/events';
import { Providers } from '@/common';

const count = (message: BaseMessage): number => Math.ceil(JSON.stringify(message.content).length / 4) + 4;

afterEach(() => jest.restoreAllMocks());

it.each([false, true])('limits actual HTTP attempts across primary and fallback transports (fallback=%s)', async hasFallback => {
  const calls: string[] = [];
  const primary = 'gpt-4.1-mini';
  const fallback = 'gpt-4.1-mini-fallback';
  const fetch = jest.fn<ReturnType<typeof globalThis.fetch>, Parameters<typeof globalThis.fetch>>(async (_input, init) => {
    if (typeof init?.body !== 'string') throw new Error('Expected a JSON request body');
    const {model} = JSON.parse(init.body) as {model: string};
    calls.push(model);
    if (calls.filter(name => name === model).length > 3) throw new DOMException('Unexpected nested retry', 'AbortError');
    return Response.json({error: {message: 'Service unavailable', type: 'server_error', code: 'server_error'}}, {
      status: 503, headers: {'retry-after': '0'},
    });
  });
  const options = {apiKey: 'offline-test', model: primary, useResponsesApi: false, maxRetries: 6, configuration: {fetch, maxRetries: 6}};
  const clientOptions = {...options, fallbacks: hasFallback ? [{provider: Providers.OPENAI,
    clientOptions: {...options, model: fallback},
  }] : []};
  const context = AgentContext.fromConfig({agentId: 'agent', provider: Providers.OPENAI,
    clientOptions, instructions: 'Checkpoint.', maxContextTokens: 4096, summarizationEnabled: true,
    summarizationConfig: {prompt: 'Checkpoint.', maxSummaryTokens: 128, retainRecent: {turns: 0}},
  }, count);
  jest.spyOn(events, 'safeDispatchCustomEvent').mockResolvedValue();
  const graph = {contentData: [] as t.RunStep[], contentIndexMap: new Map<string, number>(), isMultiAgent: false,
    dispatchRunStep: jest.fn(async (_step: t.RunStep): Promise<void> => {}),
    dispatchRunStepCompleted: jest.fn(async (): Promise<void> => {}),
    closeRunStep: jest.fn(async (): Promise<void> => {}),
  };
  const node = createSummarizeNode({agentContext: context, graph, generateStepId: () => ['summary', 0]});
  const messages = [new HumanMessage('Retain this history.')];
  const before = JSON.stringify(messages);
  const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'manual', windowMessages: messages}}, {});
  expect(calls.filter(name => name === primary)).toHaveLength(3);
  expect(calls.filter(name => name === fallback)).toHaveLength(hasFallback ? 3 : 0);
  expect(calls).toHaveLength(hasFallback ? 6 : 3);
  expect(result.messages).toBeUndefined();
  expect(JSON.stringify(messages)).toBe(before);
  expect(context.getSummaryText()).toBeUndefined();
  expect(options.maxRetries).toBe(6);
  expect(options.configuration.maxRetries).toBe(6);
});

it('disables nested default retries without mutating caller-owned options', () => {
  const openai = {maxRetries: 6, configuration: {maxRetries: 8, baseURL: 'https://offline.invalid'}};
  expect(withSummarizationRetryOwner(Providers.OPENAI, openai)).toEqual({maxRetries: 0, configuration: {...openai.configuration, maxRetries: 0}});
  const anthropic = {maxRetries: 6, clientOptions: {maxRetries: 8}};
  expect(withSummarizationRetryOwner(Providers.ANTHROPIC, anthropic)).toEqual({maxRetries: 0, clientOptions: {maxRetries: 0}});
  const bedrock = {maxRetries: 6, clientOptions: {maxAttempts: 8}};
  expect(withSummarizationRetryOwner(Providers.BEDROCK, bedrock)).toEqual({maxRetries: 0, clientOptions: {maxAttempts: 1}});
  expect(openai.maxRetries).toBe(6);
  expect(anthropic.clientOptions.maxRetries).toBe(8);
  expect(bedrock.clientOptions.maxAttempts).toBe(8);
});

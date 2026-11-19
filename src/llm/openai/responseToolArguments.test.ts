import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, expect, it } from '@jest/globals';
import { AIMessage } from '@langchain/core/messages';
import { concat } from '@langchain/core/utils/stream';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type { OpenAIClient } from '@langchain/openai';
import type { StreamLimitState } from '@/llm/streamLimits';
import {
  enforceStreamLimitsForWireChunk,
  resolveStreamLimits,
  StreamLimitExceededError,
} from '@/llm/streamLimits';
import { ChatOpenAI, AzureChatOpenAI } from './index';

type ResponseEvent = OpenAIClient.Responses.ResponseStreamEvent;
type FunctionCall = OpenAIClient.Responses.ResponseFunctionToolCall;
type ArgumentDelivery = 'deltas' | 'done' | 'item' | 'partial' | 'initial';

const calls: FunctionCall[] = [
  {
    type: 'function_call',
    id: 'fc_search_a',
    call_id: 'call_91dbfb',
    name: 'web_search',
    arguments: '{"query":"weather"}',
  },
  {
    type: 'function_call',
    id: 'fc_search_b',
    call_id: 'call_b5Pcsy',
    name: 'web_search',
    arguments: '{"query":"news"}',
  },
  {
    type: 'function_call',
    id: 'fc_articles',
    call_id: 'call_articles',
    name: 'search_articles_mcp_test',
    arguments: '{"query":"SDK streaming"}',
  },
  {
    type: 'function_call',
    id: 'fc_categories',
    call_id: 'call_categories',
    name: 'list_categories_mcp_test',
    arguments: '{"knowledgeBaseId":"kb_123"}',
  },
  {
    type: 'function_call',
    id: 'fc_search_c',
    call_id: 'call_search_c',
    name: 'web_search',
    arguments: '{"query":"parallel calls"}',
  },
];

function argumentEvents(
  deliveries: readonly ArgumentDelivery[]
): ResponseEvent[] {
  const events: ResponseEvent[] = [];
  for (const [index, call] of calls.entries()) {
    events.push({
      type: 'response.output_item.added',
      sequence_number: events.length,
      output_index: index + 1,
      item: {
        ...call,
        arguments: deliveries[index] === 'initial' ? call.arguments : '',
        status: 'in_progress',
      },
    });
  }
  for (const [index, call] of calls.entries()) {
    const delivery = deliveries[index];
    if (delivery !== 'deltas' && delivery !== 'partial') continue;
    const delta =
      delivery === 'partial' ? call.arguments.slice(0, 10) : call.arguments;
    events.push({
      type: 'response.function_call_arguments.delta',
      sequence_number: events.length,
      item_id: call.id!,
      output_index: index + 1,
      delta,
    });
  }
  for (const [index, call] of calls.entries()) {
    if (deliveries[index] !== 'item') {
      events.push({
        type: 'response.function_call_arguments.done',
        sequence_number: events.length,
        item_id: call.id!,
        output_index: index + 1,
        name: call.name,
        arguments: call.arguments,
      });
    }
    events.push({
      type: 'response.output_item.done',
      sequence_number: events.length,
      output_index: index + 1,
      item: { ...call, status: 'completed' },
    });
  }
  return events;
}

function fixture(
  events: readonly ResponseEvent[],
  provider: 'openai' | 'azure' = 'openai'
) {
  const fields = {
    apiKey: 'offline-test',
    model: 'gpt-6-luna',
    useResponsesApi: true,
    streaming: true,
    maxRetries: 0,
    configuration: {
      fetch: async (): Promise<Response> =>
        new Response(
          events
            .map(
              (event) =>
                `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`
            )
            .join(''),
          { headers: { 'content-type': 'text/event-stream' } }
        ),
      baseURL: 'https://responses-arguments.invalid/v1',
      maxRetries: 0,
    },
  };
  return provider === 'openai'
    ? new ChatOpenAI(fields)
    : new AzureChatOpenAI({
      ...fields,
      azureOpenAIApiKey: 'offline-test',
      azureOpenAIApiInstanceName: 'offline-instance',
      azureOpenAIApiDeploymentName: 'offline-deployment',
      azureOpenAIApiVersion: '2025-04-01-preview',
    });
}

async function assemble(
  events: readonly ResponseEvent[],
  provider: 'openai' | 'azure' = 'openai'
): Promise<AIMessageChunk> {
  let message: AIMessageChunk | undefined;
  for await (const chunk of await fixture(events, provider).stream(
    'Make five parallel tool calls.'
  )) {
    message = message == null ? chunk : concat(message, chunk);
  }
  if (message == null) throw new Error('Expected Responses message chunks');
  return message;
}

function expectArguments(
  message: Pick<AIMessageChunk, 'tool_calls' | 'invalid_tool_calls'>
): void {
  expect(message.invalid_tool_calls).toEqual([]);
  expect(message.tool_calls).toEqual(
    calls.map((call) => ({
      type: 'tool_call',
      id: call.call_id,
      name: call.name,
      args: JSON.parse(call.arguments),
    }))
  );
}

describe.each(['openai', 'azure'] as const)(
  '%s Responses function arguments',
  (provider) => {
    it.each(['deltas', 'done', 'item', 'partial', 'initial'] as const)(
      'keeps each parallel call’s arguments with %s delivery',
      async (delivery) => {
        expectArguments(
          await assemble(argumentEvents(calls.map(() => delivery)), provider)
        );
      }
    );

    it('keeps arguments in mixed parallel calls without duplicating terminal snapshots', async () => {
      const events = argumentEvents([
        'deltas',
        'done',
        'item',
        'partial',
        'initial',
      ]);
      expectArguments(await assemble(events, provider));
      expectArguments(
        await fixture(events, provider).invoke('Make five parallel tool calls.')
      );
    });

    it('keeps arguments when the arguments-done event is the last snapshot', async () => {
      const events = argumentEvents(calls.map(() => 'done')).filter(
        (event) => event.type !== 'response.output_item.done'
      );
      expectArguments(await assemble(events, provider));
    });

    it('recovers identity when only the completed function item is provided', async () => {
      const events = argumentEvents(calls.map(() => 'item')).filter(
        (event) => event.type === 'response.output_item.done'
      );
      expectArguments(await assemble(events, provider));
    });
  }
);

describe('Responses terminal argument safety', () => {
  it('keeps complete arguments in the generation end callback and response metadata', async () => {
    const events = argumentEvents(calls.map(() => 'done'));
    events.push({
      type: 'response.completed',
      sequence_number: events.length,
      response: {
        id: 'resp_parallel',
        created_at: 0,
        output_text: '',
        error: null,
        incomplete_details: null,
        instructions: null,
        metadata: {},
        model: 'gpt-6-luna',
        object: 'response',
        output: calls,
        parallel_tool_calls: true,
        temperature: null,
        tool_choice: 'auto',
        tools: [],
        top_p: null,
        status: 'completed',
      },
    });
    let callbackCalls: AIMessageChunk['tool_calls'];
    const message = await fixture(events).invoke(
      'Make five parallel tool calls.',
      {
        callbacks: [
          {
            handleLLMEnd(output: LLMResult): void {
              const generation = output.generations.at(0)?.at(0);
              if (
                generation != null &&
                'message' in generation &&
                AIMessage.isInstance(generation.message)
              ) {
                callbackCalls = generation.message.tool_calls;
              }
            },
          },
        ],
      }
    );
    expectArguments(message);
    expect(callbackCalls).toEqual(message.tool_calls);
    expect(message.response_metadata.output).toEqual(calls);
  });

  it.each(['arguments', 'name', 'call_id'] as const)(
    'rejects a conflicting terminal %s instead of executing altered input',
    async (field) => {
      const events = argumentEvents(calls.map(() => 'deltas')).map(
        (event): ResponseEvent => {
          if (
            event.type !== 'response.output_item.done' ||
            event.item.type !== 'function_call' ||
            event.output_index !== 1
          )
            return event;
          return { ...event, item: { ...event.item, [field]: 'conflicting' } };
        }
      );
      await expect(assemble(events)).rejects.toThrow(
        'Responses function call at output index 1 changed after streaming.'
      );
    }
  );

  it('retains malformed terminal JSON as invalid tool calls', async () => {
    const events = argumentEvents(calls.map(() => 'done')).map(
      (event): ResponseEvent => {
        if (event.type === 'response.function_call_arguments.done')
          return { ...event, arguments: 'not-json' };
        if (
          event.type === 'response.output_item.done' &&
          event.item.type === 'function_call'
        )
          return { ...event, item: { ...event.item, arguments: 'not-json' } };
        return event;
      }
    );
    const message = await assemble(events);
    expect(message.tool_calls).toEqual([]);
    expect(message.invalid_tool_calls).toHaveLength(calls.length);
    expect(message.invalid_tool_calls?.map((call) => call.args)).toEqual(
      calls.map(() => 'not-json')
    );
  });

  it.each(['done', 'item', 'partial'] as const)(
    'charges recovered %s arguments against the existing byte limit',
    async (delivery) => {
      const graph: StreamLimitState = {
        streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 18 }),
      };
      const consume = async (): Promise<void> => {
        for await (const chunk of await fixture(
          argumentEvents(calls.map(() => delivery))
        ).stream('Make five parallel tool calls.')) {
          enforceStreamLimitsForWireChunk({ graph, metadata: {}, chunk });
        }
      };
      await expect(consume()).rejects.toBeInstanceOf(StreamLimitExceededError);
    }
  );
});

describe('Responses tool schema validation', () => {
  const search = tool(async ({ query }) => query, {
    name: 'web_search',
    description: 'Search the web',
    schema: z.object({ query: z.string() }),
  });

  it('delivers terminal arguments to tools with required fields', async () => {
    const message = await assemble(argumentEvents(calls.map(() => 'done')));
    const searchCalls =
      message.tool_calls?.filter((call) => call.name === search.name) ?? [];
    expect(
      await Promise.all(searchCalls.map((call) => search.invoke(call)))
    ).toHaveLength(3);
  });

  it('still rejects a genuinely empty argument object', async () => {
    const events = argumentEvents(calls.map(() => 'done')).map(
      (event): ResponseEvent => {
        if (event.type === 'response.function_call_arguments.done')
          return { ...event, arguments: '{}' };
        if (
          event.type === 'response.output_item.done' &&
          event.item.type === 'function_call'
        )
          return { ...event, item: { ...event.item, arguments: '{}' } };
        return event;
      }
    );
    const message = await assemble(events);
    expect(message.tool_calls?.[0].args).toEqual({});
    await expect(search.invoke(message.tool_calls![0])).rejects.toThrow();
  });
});

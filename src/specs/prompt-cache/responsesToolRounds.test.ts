import { tool } from '@langchain/core/tools';
import type * as t from '@/types';
import type { Json } from './harness';
import type { JsonObject, LiveTurn } from './liveRun';
import { firstTurnMessages, restoreHistory, runLiveTurn } from './liveRun';
import { ContentTypes, GraphEvents, Providers } from '@/common';

/**
 * OpenAI Responses: a tool round restored from stored parts must serialize to
 * the input items the live run sent (item ids, status, reasoning items, no
 * empty assistant message), so every request starts with the previous one.
 */

type Reasoning = 'encrypted' | 'reference' | 'none';

interface ScriptedRound {
  text?: string;
  calls?: Array<{ name: string; query: string }>;
}

let responseCount = 0;

function frame(event: JsonObject): string {
  return `event: ${String(event.type)}\ndata: ${JSON.stringify(event)}\n\n`;
}

function usage(): JsonObject {
  return {
    input_tokens: 10,
    output_tokens: 5,
    total_tokens: 15,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens_details: { reasoning_tokens: 0 },
  };
}

/** Provider output items for one model response, then its stream frames. */
function responseStream(reasoning: Reasoning, round: ScriptedRound): string[] {
  const n = ++responseCount;
  const output: JsonObject[] = [];
  if (reasoning !== 'none') {
    output.push({
      id: `rs_${n}`,
      type: 'reasoning',
      summary: [{ type: 'summary_text', text: `Thinking ${n}.` }],
      ...(reasoning === 'encrypted' && { encrypted_content: `ENC_${n}` }),
    });
  }
  if (round.text != null) {
    output.push({
      id: `msg_${n}`,
      type: 'message',
      status: 'completed',
      content: [
        { type: 'output_text', annotations: [], logprobs: [], text: round.text },
      ],
      role: 'assistant',
    });
  }
  for (const [index, call] of (round.calls ?? []).entries()) {
    output.push({
      id: `fc_${n}_${index}`,
      type: 'function_call',
      status: 'completed',
      arguments: JSON.stringify({ query: call.query }),
      call_id: `call_${n}_${index}`,
      name: call.name,
    });
  }
  const response = {
    id: `resp_${n}`,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model: 'gpt-5',
    usage: usage(),
  };
  const frames = [
    frame({
      type: 'response.created',
      sequence_number: 0,
      response: { ...response, status: 'in_progress', output: [] },
    }),
  ];
  output.forEach((item, outputIndex) => {
    const sequence = frames.length;
    if (item.type === 'message') {
      frames.push(
        frame({
          type: 'response.output_item.added',
          sequence_number: sequence,
          output_index: outputIndex,
          item: { ...item, status: 'in_progress', content: [] },
        }),
        frame({
          type: 'response.output_text.delta',
          sequence_number: sequence + 1,
          item_id: item.id,
          output_index: outputIndex,
          content_index: 0,
          delta: round.text ?? '',
          logprobs: [],
        })
      );
    } else if (item.type === 'function_call') {
      frames.push(
        frame({
          type: 'response.output_item.added',
          sequence_number: sequence,
          output_index: outputIndex,
          item: { ...item, status: 'in_progress', arguments: '' },
        }),
        frame({
          type: 'response.function_call_arguments.delta',
          sequence_number: sequence + 1,
          item_id: item.id,
          output_index: outputIndex,
          delta: item.arguments,
        })
      );
    } else {
      frames.push(
        frame({
          type: 'response.output_item.added',
          sequence_number: sequence,
          output_index: outputIndex,
          item: { id: item.id, type: 'reasoning', summary: [] },
        })
      );
    }
    frames.push(
      frame({
        type: 'response.output_item.done',
        sequence_number: frames.length,
        output_index: outputIndex,
        item,
      })
    );
  });
  frames.push(
    frame({
      type: 'response.completed',
      sequence_number: frames.length,
      response: { ...response, status: 'completed', output },
    })
  );
  return frames;
}

const TOOL_NAMES = new Set(['lookup_order', 'search_docs']);

function responsesAgent(
  eventDriven: boolean,
  clientOptions: Record<string, Json> = {}
): t.AgentInputs {
  const definitions = [...TOOL_NAMES].map((name) => ({
    name,
    description: `Runs ${name}.`,
    parameters: {
      type: 'object' as const,
      properties: { query: { type: 'string' as const } },
      required: ['query'],
    },
  }));
  return {
    agentId: 'responses-agent',
    provider: Providers.OPENAI,
    clientOptions: {
      model: 'gpt-5',
      apiKey: 'test-key',
      maxRetries: 0,
      streaming: true,
      useResponsesApi: true,
      ...clientOptions,
    },
    instructions: 'Answer support questions.',
    maxContextTokens: 100_000,
    ...(eventDriven
      ? { toolDefinitions: definitions }
      : {
        tools: definitions.map((definition) =>
          tool(async () => `${definition.name} result`, {
            name: definition.name,
            description: definition.description,
            schema: definition.parameters,
          })
        ),
      }),
  } as unknown as t.AgentInputs;
}

const inputOf = (request: JsonObject | undefined): Json[] =>
  Array.isArray(request?.input) ? request.input : [];

/** Every request's input starts with the previous request's whole input. */
function expectStrictPrefixes(requests: JsonObject[]): void {
  for (let i = 1; i < requests.length; i++) {
    const before = JSON.stringify(inputOf(requests[i - 1])).slice(0, -1);
    const after = JSON.stringify(inputOf(requests[i]));
    expect(after.startsWith(before)).toBe(true);
  }
}

const isEmptyAssistantMessage = (item: Json): boolean =>
  item != null &&
  typeof item === 'object' &&
  !Array.isArray(item) &&
  item.role === 'assistant' &&
  (item.content === '' ||
    (Array.isArray(item.content) && item.content.length === 0));

const TURNS: Array<{ user: string; rounds: ScriptedRound[]; answer: string }> = [
  {
    user: 'Where are orders 1 and 2?',
    rounds: [
      {
        calls: [
          { name: 'lookup_order', query: '1' },
          { name: 'lookup_order', query: '2' },
        ],
      },
      { text: 'Checking the policy too.', calls: [{ name: 'search_docs', query: 'shipping' }] },
      { calls: [{ name: 'lookup_order', query: '3' }] },
    ],
    answer: 'Both ship today.',
  },
  {
    user: 'And the warranty?',
    rounds: [{ calls: [{ name: 'search_docs', query: 'warranty' }] }],
    answer: 'Two years.',
  },
  { user: 'Thanks.', rounds: [], answer: 'You are welcome.' },
];

async function runConversation(
  agent: t.AgentInputs,
  eventDriven: boolean,
  reasoning: Reasoning,
  aggregatedEvents: GraphEvents[] = [GraphEvents.ON_PROVIDER_OUTPUT_ITEMS]
): Promise<LiveTurn[]> {
  const options = { agent, toolResults: new Map(), eventDriven, aggregatedEvents };
  const turns: LiveTurn[] = [];
  const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [];
  for (const turn of TURNS) {
    exchanges.push({ user: turn.user });
    const messages =
      turns.length === 0
        ? firstTurnMessages(turn.user)
        : restoreHistory(exchanges, TOOL_NAMES);
    const streams = [
      ...turn.rounds.map((round) => responseStream(reasoning, round)),
      responseStream(reasoning, { text: turn.answer }),
    ];
    const live = await runLiveTurn(options, messages, streams);
    expect(live.requests).toHaveLength(streams.length);
    turns.push(live);
    exchanges[exchanges.length - 1].parts = live.contentParts;
  }
  return turns;
}

describe('Responses tool rounds restored across turns', () => {
  const eventDriven = true;
  it.each(['encrypted', 'reference'] as Reasoning[])(
    'keeps every request a prefix continuation with reasoning %s',
    async (reasoning) => {
      const turns = await runConversation(
        responsesAgent(eventDriven),
        eventDriven,
        reasoning
      );
      const requests = turns.flatMap((turn) => turn.requests);
      expectStrictPrefixes(requests);

      const last = inputOf(requests[requests.length - 1]);
      expect(last.some(isEmptyAssistantMessage)).toBe(false);
      const serialized = JSON.stringify(last);
      expect(serialized).toMatch(/"id":"fc_\d+_0","type":"function_call","status":"completed"/);
      expect(serialized).toMatch(/"id":"msg_\d+","type":"message","status":"completed"/);
      expect(/"id":"rs_\d+"/.test(serialized)).toBe(reasoning !== 'none');

      const stored = turns[0].contentParts.filter(
        (part): part is t.ToolCallContent => part.type === ContentTypes.TOOL_CALL
      );
      expect(stored.every((part) => part.tool_call?.providerItems != null)).toBe(
        true
      );
    }
  );
});

describe('Responses tool rounds restored without the stored items', () => {
  it('falls back to field conversion without an empty assistant message', async () => {
    const turns = await runConversation(
      responsesAgent(false),
      false,
      'encrypted',
      []
    );
    const restored = inputOf(turns[1].requests[0]);
    expect(restored.some(isEmptyAssistantMessage)).toBe(false);
    expect(JSON.stringify(restored)).not.toContain('"id":"fc_');
  });
});

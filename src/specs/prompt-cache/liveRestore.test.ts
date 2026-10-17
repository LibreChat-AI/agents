import { tool } from '@langchain/core/tools';
import type * as t from '@/types';
import type { Json } from './harness';
import type {
  Protocol,
  ScriptedReply,
  ScriptedReasoning,
} from './scriptedReplies';
import type { LiveTurn, ToolOutput } from './liveRun';
import { restoreHistory, runLiveTurn } from './liveRun';
import type { ProviderCase } from './providers';
import { PROVIDER_CASES } from './providers';
import { scriptReply } from './scriptedReplies';
import { GraphEvents, Providers } from '@/common';

/**
 * The strongest cache invariant: a turn restored from the parts a host
 * stored must serialize exactly like the live requests of that turn. For
 * every provider client, the first request of turn N+1 starts, segment by
 * segment and byte for byte, with the last request of turn N. Only cache
 * markers, which move to the newest turn by design, are set aside.
 */

jest.setTimeout(120_000);

const PROTOCOLS: Record<string, Protocol> = {
  openAI: 'chat',
  'openAI responses': 'responses',
  'openAI-compatible endpoint': 'chat',
  azureOpenAI: 'chat',
  deepseek: 'chat',
  xai: 'chat',
  moonshot: 'chat',
  anthropic: 'anthropic',
  'anthropic promptCache': 'anthropic',
  openrouter: 'chat',
  'openrouter promptCache': 'chat',
  bedrock: 'bedrock',
  'bedrock promptCache': 'bedrock',
  google: 'google',
};

const PIXEL_PNG =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

const TOOL_OUTPUTS = new Map<string, ToolOutput>([
  ['lookup_order', 'Order 42: packed, ships tomorrow'],
  ['search_docs', 'Returns: thirty days'],
  [
    'take_screenshot',
    [
      { type: 'text', text: 'Screenshot of the order page' },
      { type: 'image_url', image_url: { url: `data:image/png;base64,${PIXEL_PNG}` } },
    ] as t.MessageContentComplex[],
  ],
  [
    'read_page',
    [
      { type: 'text', text: 'Shipping page' },
      { type: 'text', text: 'Orders ship within two days.' },
    ] as t.MessageContentComplex[],
  ],
]);

interface TurnScript {
  user: string;
  replies: ScriptedReply[];
}

const TURNS: TurnScript[] = [
  {
    user: 'Where are orders 41 and 42?',
    replies: [
      {
        calls: [
          { name: 'lookup_order', args: { query: '41' } },
          { name: 'lookup_order', args: { query: '42' } },
        ],
      },
      {
        text: 'Checking the policy too.',
        calls: [{ name: 'search_docs', args: { query: 'returns' } }],
      },
      { text: 'Both ship tomorrow; returns within thirty days.' },
    ],
  },
  {
    user: 'What does the shipping page say?',
    replies: [
      {
        calls: [
          { name: 'read_page', args: { query: 'shipping' } },
          { name: 'search_docs', args: { query: 'shipping' } },
        ],
      },
      { text: 'Orders ship within two days.' },
    ],
  },
  {
    user: 'Show me the order page.',
    replies: [
      { text: '', calls: [{ name: 'take_screenshot', args: { query: 'order' } }] },
      { text: 'Here it is.' },
    ],
  },
  { user: 'Thanks.', replies: [{ text: 'You are welcome.' }] },
];

function definitions(): t.LCTool[] {
  return [...TOOL_OUTPUTS.keys()].map((name) => ({
    name,
    description: `Runs ${name}.`,
    parameters: {
      type: 'object',
      properties: { query: { type: 'string' } },
      required: ['query'],
    },
  }));
}

function agentFor(
  clientOptions: t.ClientOptions,
  provider: t.AgentInputs['provider'],
  eventDriven: boolean
): t.AgentInputs {
  const tools = definitions();
  return {
    agentId: 'live-restore-agent',
    provider,
    clientOptions: { ...clientOptions, streaming: true },
    instructions: 'Answer support questions.',
    maxContextTokens: 100_000,
    ...(eventDriven
      ? { toolDefinitions: tools }
      : {
        tools: tools.map((definition) =>
          tool(async () => TOOL_OUTPUTS.get(definition.name) ?? '', {
            name: definition.name,
            description: definition.description ?? '',
            schema: definition.parameters as Record<string, unknown>,
          })
        ),
      }),
  } as t.AgentInputs;
}

function isObject(value: Json): value is { [key: string]: Json } {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/** Drops cache markers, which move to the newest turn by design. */
function withoutCacheMarkers(value: Json): Json {
  if (Array.isArray(value)) {
    return value
      .filter((item) => !(isObject(item) && Object.keys(item).length === 1 && 'cachePoint' in item))
      .map((item) => withoutCacheMarkers(item));
  }
  if (!isObject(value)) {
    return value;
  }
  const entries: [string, Json][] = [];
  for (const [key, nested] of Object.entries(value)) {
    if (key === 'cache_control') {
      continue;
    }
    entries.push([
      key,
      withoutCacheMarkers(nested),
    ]);
  }
  return Object.fromEntries(entries);
}

function segments(
  reduce: (body: Json) => { tools: Json[]; system: Json; messages: Json[] },
  body: Json,
  marked: boolean
): string[] {
  const request = reduce(body);
  const normalize = (value: Json): string =>
    JSON.stringify(marked ? withoutCacheMarkers(value) : value);
  return [
    ...request.tools.map((segment) => normalize(segment)),
    normalize(request.system),
    ...request.messages.map((message) => normalize(message)),
  ];
}

const thinkingCase = (
  name: string,
  base: string,
  extra: Record<string, Json>
): ProviderCase => {
  const providerCase = PROVIDER_CASES.find((candidate) => candidate.name === base);
  if (providerCase == null) {
    throw new Error(`Unknown provider case ${base}`);
  }
  return {
    ...providerCase,
    name,
    clientOptions: () =>
      ({ ...providerCase.clientOptions(), ...extra }) as t.ClientOptions,
  };
};

const ANTHROPIC_THINKING = { thinking: { type: 'enabled', budget_tokens: 1024 }, maxTokens: 4096 };
const BEDROCK_THINKING = {
  additionalModelRequestFields: { thinking: { type: 'enabled', budget_tokens: 1024 } },
  maxTokens: 4096,
};

/**
 * Regression cases for reasoning the provider requires back on tool rounds:
 * signed `thinking` (including empty text) and `redacted_thinking` must reach
 * the follow-up request of their turn unchanged. Restored history does not
 * replay reasoning of earlier turns (unchanged SDK behavior), so the turn
 * boundary is compared with those blocks set aside.
 */
const THINKING_CASES: Array<[ProviderCase, ScriptedReasoning[]]> = [
  [
    thinkingCase('anthropic thinking', 'anthropic', ANTHROPIC_THINKING),
    [{ text: 'Reasoning.', signature: 'SIG_SIGNED' }],
  ],
  [
    thinkingCase('anthropic thinking redacted', 'anthropic', ANTHROPIC_THINKING),
    [{ redacted: 'REDACTED_DATA' }, { text: '', signature: 'SIG_EMPTY_TEXT' }],
  ],
  [
    thinkingCase('anthropic promptCache thinking', 'anthropic promptCache', ANTHROPIC_THINKING),
    [{ text: 'Reasoning.', signature: 'SIG_SIGNED' }],
  ],
  [
    thinkingCase('bedrock thinking', 'bedrock', BEDROCK_THINKING),
    [{ text: 'Reasoning.', signature: 'SIG_SIGNED' }],
  ],
];

const cases: Array<[ProviderCase, ScriptedReasoning[] | undefined]> = [
  ...PROVIDER_CASES.filter((providerCase) => providerCase.name in PROTOCOLS).map(
    (providerCase): [ProviderCase, undefined] => [providerCase, undefined]
  ),
  ...THINKING_CASES,
];

const REASONING_BLOCK = /\{"type":"(?:thinking|redacted_thinking)"[^{}]*\},?|\{"reasoningContent":\{"reasoningText":\{[^{}]*\}\}\},?/g;

const withoutReasoning = (segment: string): string =>
  segment.replace(REASONING_BLOCK, '');

describe.each(cases.map(([providerCase, reasoning]) => [providerCase.name, providerCase, reasoning] as const))(
  'live and restored requests: %s',
  (_name, providerCase, reasoning) => {
    const protocol =
      PROTOCOLS[providerCase.name] ??
      (providerCase.provider === Providers.BEDROCK ? 'bedrock' : 'anthropic');
    const marked = providerCase.breakpoints != null;

    it('turn N+1 starts with the bytes of turn N', async () => {
      const eventDriven = true;
      const turns: LiveTurn[] = [];
      const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [];
      for (const turn of TURNS) {
        exchanges.push({ user: turn.user });
        const live = await runLiveTurn(
          {
            agent: agentFor(providerCase.clientOptions(), providerCase.provider, eventDriven),
            toolResults: TOOL_OUTPUTS,
            eventDriven,
            transport: protocol === 'bedrock' ? 'bedrock' : 'fetch',
            aggregatedEvents: [GraphEvents.ON_PROVIDER_OUTPUT_ITEMS],
          },
          restoreHistory(exchanges, new Set(TOOL_OUTPUTS.keys())),
          turn.replies.map((reply) =>
            scriptReply(protocol, reasoning != null ? { ...reply, reasoning } : reply)
          )
        );
        expect(live.requests).toHaveLength(turn.replies.length);
        exchanges[exchanges.length - 1].parts = live.contentParts;
        turns.push(live);
      }
      for (const { requests } of turns) {
        for (let i = 1; i < requests.length; i++) {
          const before = segments(providerCase.reduce, requests[i - 1], marked);
          const after = segments(providerCase.reduce, requests[i], marked);
          expect(after.slice(0, before.length)).toEqual(before);
          const sent = after.slice(before.length).join('\n');
          for (const block of reasoning ?? []) {
            expect(sent).toContain(
              'redacted' in block
                ? `"data":"${block.redacted}"`
                : `"signature":"${block.signature}"`
            );
          }
        }
      }
      const comparable = (segment: string): string =>
        reasoning != null ? withoutReasoning(segment) : segment;
      for (let i = 1; i < turns.length; i++) {
        const previous = turns[i - 1].requests;
        const before = segments(providerCase.reduce, previous[previous.length - 1], marked).map(comparable);
        const after = segments(providerCase.reduce, turns[i].requests[0], marked).map(comparable);
        expect(after.slice(0, before.length)).toEqual(before);
      }
    });
  }
);

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
 * Reasoning the provider requires back on tool rounds: signed `thinking`
 * (including empty text) and `redacted_thinking` must reach the follow-up
 * request of their turn unchanged. Every scripted reply gets its own
 * signatures, so a block restored from the wrong reply or position shows up.
 *
 * `keepsPriorThinking` cases run the host side of reasoning replay (the agent
 * captures `provider_replay`, history is restored with
 * `replayProviderReasoning`) and must keep earlier turns' thinking blocks
 * byte for byte at the turn boundary. The other cases are the baseline
 * without replay, which sets those blocks aside.
 */
interface ThinkingCase {
  providerCase: ProviderCase;
  reasoning: ScriptedReasoning[];
  /** Reasoning emitted after the text of a reply (interleaved thinking). */
  reasoningAfterText?: ScriptedReasoning[];
  keepsPriorThinking: boolean;
}

const SIGNED = [{ text: 'Reasoning.', signature: 'SIG_SIGNED' }];
const REDACTED_AND_EMPTY = [
  { redacted: 'REDACTED_DATA' },
  { text: '', signature: 'SIG_EMPTY_TEXT' },
];
const MID_REPLY = [{ text: 'Reasoning after the text.', signature: 'SIG_AFTER' }];

const THINKING_CASES: ThinkingCase[] = [
  { providerCase: thinkingCase('anthropic thinking', 'anthropic', ANTHROPIC_THINKING), reasoning: SIGNED, keepsPriorThinking: false },
  { providerCase: thinkingCase('anthropic thinking redacted', 'anthropic', ANTHROPIC_THINKING), reasoning: REDACTED_AND_EMPTY, keepsPriorThinking: false },
  { providerCase: thinkingCase('anthropic promptCache thinking', 'anthropic promptCache', ANTHROPIC_THINKING), reasoning: SIGNED, keepsPriorThinking: false },
  { providerCase: thinkingCase('bedrock thinking', 'bedrock', BEDROCK_THINKING), reasoning: SIGNED, keepsPriorThinking: false },
  { providerCase: thinkingCase('anthropic thinking replay', 'anthropic', ANTHROPIC_THINKING), reasoning: SIGNED, keepsPriorThinking: true },
  { providerCase: thinkingCase('anthropic thinking redacted replay', 'anthropic', ANTHROPIC_THINKING), reasoning: REDACTED_AND_EMPTY, keepsPriorThinking: true },
  { providerCase: thinkingCase('anthropic thinking interleaved replay', 'anthropic', ANTHROPIC_THINKING), reasoning: SIGNED, reasoningAfterText: MID_REPLY, keepsPriorThinking: true },
  { providerCase: thinkingCase('anthropic promptCache thinking replay', 'anthropic promptCache', ANTHROPIC_THINKING), reasoning: SIGNED, keepsPriorThinking: true },
  { providerCase: thinkingCase('bedrock thinking replay', 'bedrock', BEDROCK_THINKING), reasoning: SIGNED, keepsPriorThinking: true },
  { providerCase: thinkingCase('bedrock thinking interleaved replay', 'bedrock', BEDROCK_THINKING), reasoning: SIGNED, reasoningAfterText: MID_REPLY, keepsPriorThinking: true },
];

const cases: ThinkingCase[] = [
  ...PROVIDER_CASES.filter((providerCase) => providerCase.name in PROTOCOLS).map(
    (providerCase): ThinkingCase => ({
      providerCase,
      reasoning: [],
      keepsPriorThinking: false,
    })
  ),
  ...THINKING_CASES,
];

const REASONING_BLOCK = /\{"type":"(?:thinking|redacted_thinking)"[^{}]*\},?|\{"reasoningContent":\{"reasoningText":\{[^{}]*\}\}\},?/g;

const withoutReasoning = (segment: string): string =>
  segment.replace(REASONING_BLOCK, '');

const protocolOf = (providerCase: ProviderCase): Protocol =>
  PROTOCOLS[providerCase.name] ??
  (providerCase.provider === Providers.BEDROCK ? 'bedrock' : 'anthropic');

/** Signatures and opaque reasoning data of a request body, in wire order. */
const reasoningMarkers = (body: Json): string[] =>
  JSON.stringify(body).match(/"(?:signature|data)":"[A-Z_]+_\d+"/g) ?? [];

let signatureSerial = 0;

/** Gives every reply of a scripted turn its own signatures. */
function withReasoning(
  reply: ScriptedReply,
  reasoning: ScriptedReasoning[],
  afterText: ScriptedReasoning[] | undefined
): { reply: ScriptedReply; markers: string[] } {
  const serial = ++signatureSerial;
  const markers: string[] = [];
  const unique = (block: ScriptedReasoning): ScriptedReasoning => {
    if ('redacted' in block) {
      markers.push(`"data":"${block.redacted}_${serial}"`);
      return { redacted: `${block.redacted}_${serial}` };
    }
    markers.push(`"signature":"${block.signature}_${serial}"`);
    return { ...block, signature: `${block.signature}_${serial}` };
  };
  if (reasoning.length === 0) {
    return { reply, markers };
  }
  const isCallingText = (reply.text ?? '') !== '' && (reply.calls?.length ?? 0) > 0;
  return {
    reply: {
      ...reply,
      reasoning: reasoning.map(unique),
      ...(afterText != null && isCallingText
        ? { reasoningAfterText: afterText.map(unique) }
        : {}),
    },
    markers,
  };
}

interface PlayedTurns {
  turns: LiveTurn[];
  /** Per turn and request: markers of the replies sent before it. */
  expectedMarkers: string[][][];
}

async function playTurns(
  testCase: ThinkingCase,
  replay: boolean
): Promise<PlayedTurns> {
  const { providerCase, reasoning, reasoningAfterText } = testCase;
  const protocol = protocolOf(providerCase);
  const model = (providerCase.clientOptions() as { model: string }).model;
  const names = new Set(TOOL_OUTPUTS.keys());
  const turns: LiveTurn[] = [];
  const expectedMarkers: string[][][] = [];
  const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [];
  for (const turn of TURNS) {
    exchanges.push({ user: turn.user });
    const scripted = turn.replies.map((reply) =>
      withReasoning(reply, reasoning, reasoningAfterText)
    );
    const live = await runLiveTurn(
      {
        agent: {
          ...agentFor(providerCase.clientOptions(), providerCase.provider, true),
          ...(replay ? { providerReasoningReplay: true } : {}),
        },
        toolResults: TOOL_OUTPUTS,
        eventDriven: true,
        transport: protocol === 'bedrock' ? 'bedrock' : 'fetch',
        aggregatedEvents: [GraphEvents.ON_PROVIDER_OUTPUT_ITEMS],
      },
      restoreHistory(
        exchanges,
        names,
        replay ? { provider: providerCase.provider, model } : undefined
      ),
      scripted.map(({ reply }) => scriptReply(protocol, reply))
    );
    expect(live.requests).toHaveLength(turn.replies.length);
    exchanges[exchanges.length - 1].parts = live.contentParts;
    turns.push(live);
    expectedMarkers.push(
      scripted.map((_, index) =>
        scripted.slice(0, index).flatMap(({ markers }) => markers)
      )
    );
  }
  return { turns, expectedMarkers };
}

const boundaryPairs = (
  providerCase: ProviderCase,
  turns: LiveTurn[],
  comparable: (segment: string) => string
): Array<{ before: string[]; after: string[] }> => {
  const marked = providerCase.breakpoints != null;
  return turns.slice(1).map((turn, index) => {
    const previous = turns[index].requests;
    return {
      before: segments(providerCase.reduce, previous[previous.length - 1], marked).map(comparable),
      after: segments(providerCase.reduce, turn.requests[0], marked).map(comparable),
    };
  });
};

describe.each(cases.map((testCase) => [testCase.providerCase.name, testCase] as const))(
  'live and restored requests: %s',
  (_name, testCase) => {
    const { providerCase, reasoning, keepsPriorThinking } = testCase;
    const marked = providerCase.breakpoints != null;

    it('turn N+1 starts with the bytes of turn N', async () => {
      const { turns, expectedMarkers } = await playTurns(testCase, keepsPriorThinking);
      turns.forEach(({ requests }, turnIndex) => {
        for (let i = 1; i < requests.length; i++) {
          const before = segments(providerCase.reduce, requests[i - 1], marked);
          const after = segments(providerCase.reduce, requests[i], marked);
          expect(after.slice(0, before.length)).toEqual(before);
          const sent = JSON.stringify(requests[i]);
          for (const marker of expectedMarkers[turnIndex][i]) {
            expect(sent).toContain(marker);
          }
        }
      });
      if (keepsPriorThinking) {
        /** Append-only: no thinking block is ever removed from the middle. */
        const all = turns.flatMap(({ requests }) => requests);
        for (let i = 1; i < all.length; i++) {
          const before = reasoningMarkers(all[i - 1]);
          expect(reasoningMarkers(all[i]).slice(0, before.length)).toEqual(before);
        }
        expect(reasoningMarkers(all[all.length - 1]).length).toBeGreaterThan(
          reasoningMarkers(all[0]).length
        );
      }
      const comparable = (segment: string): string =>
        reasoning.length > 0 && !keepsPriorThinking
          ? withoutReasoning(segment)
          : segment;
      for (const { before, after } of boundaryPairs(providerCase, turns, comparable)) {
        expect(after.slice(0, before.length)).toEqual(before);
      }
    });

    if (keepsPriorThinking) {
      it('carries earlier turns thinking in the restored history', async () => {
        const { turns, expectedMarkers } = await playTurns(testCase, true);
        const lastOfFirstTurn = JSON.stringify(turns[0].requests[turns[0].requests.length - 1]);
        const firstOfSecondTurn = JSON.stringify(turns[1].requests[0]);
        for (const marker of expectedMarkers[0][expectedMarkers[0].length - 1]) {
          expect(lastOfFirstTurn).toContain(marker);
          expect(firstOfSecondTurn).toContain(marker);
        }
      });

      /** Reverting replay (capture, restore or the agent flag) breaks the boundary. */
      it('breaks the turn boundary when replay is off', async () => {
        const { turns } = await playTurns(testCase, false);
        const full = boundaryPairs(providerCase, turns, (segment) => segment);
        expect(
          full.some(({ before, after }) => {
            const head = after.slice(0, before.length);
            return JSON.stringify(head) !== JSON.stringify(before);
          })
        ).toBe(true);
        for (const { before, after } of boundaryPairs(providerCase, turns, withoutReasoning)) {
          expect(after.slice(0, before.length)).toEqual(before);
        }
      });
    }
  }
);

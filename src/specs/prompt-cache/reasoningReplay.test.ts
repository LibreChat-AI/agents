import type * as t from '@/types';
import type { LiveTurn } from './liveRun';
import { restoreHistory, runLiveTurn } from './liveRun';
import { GraphEvents, Providers } from '@/common';
import { scriptReply } from './scriptedReplies';

/**
 * Reasoning replay across model boundaries: blocks are signed by the model
 * that served the call, so a restored history only carries them back to that
 * model, and a fallback that served a call owns the blocks of that call.
 */

jest.setTimeout(120_000);

const PRIMARY = 'claude-primary-test';
const OTHER = 'claude-other-test';
const FALLBACK = 'claude-fallback-test';

const agentFor = (model: string, fallbackModel?: string): t.AgentInputs =>
  ({
    agentId: 'replay-agent',
    provider: Providers.ANTHROPIC,
    clientOptions: {
      model,
      apiKey: 'test-key',
      maxRetries: 0,
      streaming: true,
      thinking: { type: 'enabled', budget_tokens: 1024 },
      maxTokens: 4096,
      ...(fallbackModel != null && {
        fallbacks: [
          {
            provider: Providers.ANTHROPIC,
            clientOptions: { model: fallbackModel, apiKey: 'test-key', maxRetries: 0 },
          },
        ],
      }),
    },
    instructions: 'Answer support questions.',
    maxContextTokens: 100_000,
    providerReasoningReplay: true,
    toolDefinitions: [
      {
        name: 'lookup',
        description: 'Runs lookup.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
        },
      },
    ],
  }) as t.AgentInputs;

const tools = new Map<string, string>([['lookup', 'found']]);

async function turn(
  exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }>,
  model: string,
  replies: unknown[],
  options: { replayModel?: string; fallbackModel?: string } = {}
): Promise<LiveTurn> {
  const live = await runLiveTurn(
    {
      agent: agentFor(model, options.fallbackModel),
      toolResults: tools,
      eventDriven: true,
      aggregatedEvents: [GraphEvents.ON_PROVIDER_OUTPUT_ITEMS],
    },
    restoreHistory(exchanges, new Set(tools.keys()), {
      provider: Providers.ANTHROPIC,
      model: options.replayModel ?? model,
    }),
    replies
  );
  exchanges[exchanges.length - 1].parts = live.contentParts;
  return live;
}

const toolRound = (signature: string): unknown[] => [
  scriptReply('anthropic', {
    reasoning: [{ text: 'Looking up.', signature }],
    calls: [{ name: 'lookup', args: { query: 'x' } }],
  }),
  scriptReply('anthropic', {
    reasoning: [{ text: 'Answering.', signature: `${signature}_FINAL` }],
    text: 'Done.',
  }),
];

const body = (live: LiveTurn, index: number): string =>
  JSON.stringify(live.requests[index]);

describe('reasoning replay across models', () => {
  it('replays only to the model that signed the blocks, and not across another model\'s turn', async () => {
    const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [];
    exchanges.push({ user: 'one' });
    await turn(exchanges, PRIMARY, toolRound('SIG_A'));
    exchanges.push({ user: 'two' });
    const second = await turn(exchanges, OTHER, toolRound('SIG_B'));
    exchanges.push({ user: 'three' });
    const third = await turn(exchanges, PRIMARY, toolRound('SIG_C'));

    /** The other model never receives the primary model's signed blocks. */
    expect(body(second, 0)).not.toContain('SIG_A');
    expect(body(second, 1)).toContain('SIG_B');
    expect(body(second, 1)).not.toContain('SIG_A');
    /** Its turn sits between the primary model's two turns: replaying the first
     *  turn's blocks around that hole would break the chain they are bound to. */
    expect(body(third, 0)).not.toContain('SIG_A');
    expect(body(third, 0)).not.toContain('SIG_B');
    expect(third.requests).toHaveLength(2);
    expect(body(third, 1)).toContain('SIG_C');
  });

  it('replays the turns after a model change once the model has signed them', async () => {
    const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [];
    exchanges.push({ user: 'one' });
    await turn(exchanges, PRIMARY, toolRound('SIG_A'));
    exchanges.push({ user: 'two' });
    await turn(exchanges, OTHER, toolRound('SIG_B'));
    exchanges.push({ user: 'three' });
    await turn(exchanges, OTHER, toolRound('SIG_C'));
    exchanges.push({ user: 'four' });
    const fourth = await turn(exchanges, OTHER, toolRound('SIG_D'));

    expect(body(fourth, 0)).not.toContain('SIG_A');
    expect(body(fourth, 0)).toContain('SIG_B');
    expect(body(fourth, 0)).toContain('SIG_C');
  });

  it('attributes the blocks of a call served by a fallback to the fallback model', async () => {
    const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [
      { user: 'one' },
    ];
    const first = await turn(
      exchanges,
      PRIMARY,
      [
        { failWithStatus: 500 },
        scriptReply('anthropic', {
          reasoning: [{ text: 'Fallback thinking.', signature: 'SIG_FB' }],
          text: 'Served by the fallback.',
        }),
      ],
      { fallbackModel: FALLBACK }
    );
    expect(first.requests).toHaveLength(2);
    const replay = (
      first.contentParts.find((part) => part.type === 'think') as
        | t.ReasoningContentText
        | undefined
    )?.provider_replay;
    expect(replay?.anthropic?.model).toBe(FALLBACK);

    exchanges.push({ user: 'two' });
    const toPrimary = await turn(exchanges, PRIMARY, [
      scriptReply('anthropic', { text: 'ok' }),
    ]);
    expect(body(toPrimary, 0)).not.toContain('SIG_FB');

    const toFallbackModel = await turn(
      exchanges,
      FALLBACK,
      [scriptReply('anthropic', { text: 'ok' })],
    );
    expect(body(toFallbackModel, 0)).toContain('SIG_FB');
  });
});

describe('Bedrock rejecting replayed reasoning', () => {
  const BEDROCK_MODEL = 'anthropic.claude-test-v1:0';

  const bedrockTurn = async (
    exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }>,
    replies: unknown[]
  ): Promise<LiveTurn> => {
    const live = await runLiveTurn(
      {
        agent: {
          agentId: 'bedrock-replay-agent',
          provider: Providers.BEDROCK,
          clientOptions: {
            model: BEDROCK_MODEL,
            region: 'us-east-1',
            credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
            maxRetries: 0,
            streaming: true,
            maxTokens: 4096,
            additionalModelRequestFields: {
              thinking: { type: 'enabled', budget_tokens: 1024 },
            },
          },
          instructions: 'Answer support questions.',
          maxContextTokens: 100_000,
          providerReasoningReplay: true,
        } as t.AgentInputs,
        toolResults: tools,
        eventDriven: true,
        transport: 'bedrock',
      },
      restoreHistory(exchanges, new Set(tools.keys()), {
        provider: Providers.BEDROCK,
        model: BEDROCK_MODEL,
      }),
      replies
    );
    exchanges[exchanges.length - 1].parts = live.contentParts;
    return live;
  };

  const reply = (signature: string): unknown =>
    scriptReply('bedrock', {
      reasoning: [{ text: 'Thinking.', signature }],
      text: 'Answer.',
    });

  it('retries once without the signed blocks and warns once', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [
        { user: 'one' },
      ];
      await bedrockTurn(exchanges, [reply('SIG_B1')]);
      exchanges.push({ user: 'two' });
      const second = await bedrockTurn(exchanges, [
        {
          failWithStatus: 400,
          message: 'The signature in the thinking block is not valid for this conversation',
        },
        reply('SIG_B2'),
      ]);

      expect(second.requests).toHaveLength(2);
      expect(body(second, 0)).toContain('SIG_B1');
      expect(body(second, 1)).not.toContain('SIG_B1');
      expect(body(second, 1)).toContain('Answer.');
      expect(
        warn.mock.calls.filter(([message]) => String(message).includes('[bedrock]'))
      ).toHaveLength(1);
      expect(JSON.stringify(second.contentParts)).toContain('SIG_B2');
    } finally {
      warn.mockRestore();
    }
  });

  it('does not retry a rejection that is not about reasoning', async () => {
    const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [
      { user: 'one' },
    ];
    await bedrockTurn(exchanges, [reply('SIG_B1')]);
    exchanges.push({ user: 'two' });
    const second = await bedrockTurn(exchanges, [
      { failWithStatus: 400, message: 'Malformed input request' },
      reply('SIG_B2'),
    ]);

    expect(second.requests).toHaveLength(1);
  });

  it('retries only once', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }> = [
        { user: 'one' },
      ];
      await bedrockTurn(exchanges, [reply('SIG_B1')]);
      exchanges.push({ user: 'two' });
      const rejected = {
        failWithStatus: 400,
        message: 'thinking signature rejected',
      };
      const second = await bedrockTurn(exchanges, [rejected, rejected, reply('SIG_B2')]);

      expect(second.requests).toHaveLength(2);
    } finally {
      warn.mockRestore();
    }
  });
});

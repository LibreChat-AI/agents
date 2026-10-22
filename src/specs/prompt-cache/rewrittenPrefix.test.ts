import type * as t from '@/types';
import { restoreHistory, runLiveTurn } from './liveRun';
import { GraphEvents, Providers } from '@/common';
import { scriptReply } from './scriptedReplies';

/**
 * A signed block is bound to the prefix its request sent. When the context
 * management rewrites an earlier tool result between the requests of a run,
 * the turns produced before the rewrite lose their blocks, and the host is
 * told, so the history it restores next turn equals what was sent.
 */

jest.setTimeout(120_000);

const MODEL = 'claude-primary-test';

const agent = {
  agentId: 'rewrite-agent',
  provider: Providers.ANTHROPIC,
  clientOptions: {
    model: MODEL,
    apiKey: 'test-key',
    maxRetries: 0,
    streaming: true,
    thinking: { type: 'enabled', budget_tokens: 1024 },
    maxTokens: 4096,
  },
  instructions: 'Answer support questions.',
  maxContextTokens: 3000,
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
} as t.AgentInputs;

const bulky = Array.from({ length: 400 }, (_, i) => `row ${i} value ${i * 7}`).join('\n');

const lookup = (signature: string): unknown =>
  scriptReply('anthropic', {
    reasoning: [{ text: 'Looking up.', signature }],
    calls: [{ name: 'lookup', args: { query: 'x' } }],
  });

describe('a run whose earlier tool results are rewritten', () => {
  it('sends the later requests without the blocks bound to the old prefix, and withdraws them', async () => {
    const live = await runLiveTurn(
      {
        agent,
        toolResults: new Map([['lookup', bulky]]),
        eventDriven: true,
        contextManagement: true,
        aggregatedEvents: [GraphEvents.ON_PROVIDER_OUTPUT_ITEMS],
      },
      restoreHistory([{ user: 'look it up' }], new Set(['lookup']), {
        provider: Providers.ANTHROPIC,
        model: MODEL,
      }),
      [
        lookup('SIG_1'),
        lookup('SIG_2'),
        lookup('SIG_3'),
        lookup('SIG_3B'),
        lookup('SIG_3C'),
        lookup('SIG_3D'),
        scriptReply('anthropic', {
          reasoning: [{ text: 'Answering.', signature: 'SIG_4' }],
          text: 'Done.',
        }),
      ]
    );
    const bodies = live.requests.map((request) => JSON.stringify(request));
    const last = bodies[bodies.length - 1];
    const stored = JSON.stringify(
      live.contentParts.map((part) => (part as t.ReasoningContentText).provider_replay ?? null)
    );
    expect(bodies).toHaveLength(7);
    /** The result after SIG_1 shrank before the third request: SIG_2 was signed over the old one. */
    expect(bodies[2]).toContain('SIG_1');
    expect(bodies[2]).not.toContain('SIG_2');
    expect(last).not.toContain('SIG_2');
    expect(last).toContain('SIG_1');
    expect(last).toContain('SIG_3D');
    expect((last.match(/"tool_use"/g) ?? []).length).toBe(6);
    /** What the host stores equals what the requests carried. */
    expect(stored).not.toContain('SIG_2');
    expect(stored).toContain('SIG_1');
    expect(stored).toContain('SIG_3');

    /** The restored history keeps the blocks around the withdrawn one, as the run sent them. */
    const next = await runLiveTurn(
      {
        agent,
        toolResults: new Map([['lookup', bulky]]),
        eventDriven: true,
        aggregatedEvents: [GraphEvents.ON_PROVIDER_OUTPUT_ITEMS],
      },
      restoreHistory(
        [{ user: 'look it up', parts: live.contentParts }, { user: 'and again' }],
        new Set(['lookup']),
        { provider: Providers.ANTHROPIC, model: MODEL }
      ),
      [scriptReply('anthropic', { text: 'ok' })]
    );
    const restored = JSON.stringify(next.requests[0]);
    expect(restored).toContain('SIG_1');
    expect(restored).not.toContain('SIG_2');
    expect(restored).toContain('SIG_3D');
  });
});

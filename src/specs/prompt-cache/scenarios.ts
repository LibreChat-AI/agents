import type { Scenario, ToolSpec, TurnSpec } from './harness';

const STABLE_INSTRUCTIONS =
  'You are a careful support assistant. Answer in the language of the user.';

const TOOLS: ToolSpec[] = [
  { name: 'lookup_order', description: 'Looks up an order by its identifier.' },
  { name: 'search_docs', description: 'Searches the product documentation.' },
];

const CONVERSATION_A_CONTEXT =
  '# Conversation context\nConversation started: 2026-03-09 23:00 +03:00 (Mon)';
const CONVERSATION_B_CONTEXT =
  '# Conversation context\nConversation started: 2026-04-02 10:15 +03:00 (Thu)';

const withRetrievedContext = (turn: TurnSpec, index: number): TurnSpec => ({
  ...turn,
  additionalInstructions: `# Retrieved for this question\nPassage ${index + 1}: ${turn.user}`,
});

/** Compaction inside the third turn's run, with tool rounds after it. */
const IN_RUN_COMPACTION_TURNS: TurnSpec[] = [
  { user: 'Start a refund for order 9.', answer: 'Started.' },
  {
    user: 'What is its status?',
    answer: 'Pending approval.',
    toolRounds: [
      { name: 'lookup_order', args: { query: '9' }, result: 'Order 9: refund pending' },
    ],
  },
  {
    user: 'Check orders 9 and 10 and the refund policy.',
    answer: 'Both are pending; refunds take five days.',
    toolRounds: [
      { name: 'lookup_order', args: { query: '9' }, result: 'Order 9: refund pending' },
      { name: 'lookup_order', args: { query: '10' }, result: 'Order 10: refund pending' },
      { name: 'search_docs', args: { query: 'refund time' }, result: 'Refunds: five days' },
    ],
    compaction: {
      summary: 'The user started a refund for order 9; it is pending approval.',
      retainFromTurn: 1,
      afterRound: 1,
    },
  },
  { user: 'Any update?', answer: 'Not yet.' },
  {
    user: 'Look again, please.',
    answer: 'Approved now.',
    toolRounds: [
      { name: 'lookup_order', args: { query: '9' }, result: 'Order 9: refund approved' },
    ],
  },
];

/**
 * Scenarios replayed for every provider case. Turn texts stay short: the
 * invariants are about byte positions, not about prompt size.
 */
export const SCENARIOS: Scenario[] = [
  {
    name: 'chat with tool rounds and a second conversation',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: [
          { user: 'Where is my order 42?', answer: 'Let me check.' },
          {
            user: 'Please look it up.',
            answer: 'Order 42 ships tomorrow.',
            toolRounds: [
              {
                name: 'lookup_order',
                args: { query: '42' },
                result: 'Order 42: packed, ships tomorrow',
              },
            ],
          },
          { user: 'And the return policy?', answer: 'Thirty days.' },
          { user: 'Thanks.', answer: 'You are welcome.' },
        ],
      },
      {
        conversationInstructions: CONVERSATION_B_CONTEXT,
        turns: [
          { user: 'How do I reset my password?', answer: 'Use the link.' },
          { user: 'Which link?', answer: 'The one on the sign-in page.' },
        ],
      },
    ],
  },
  {
    name: 'per-turn volatile context beside a conversation block',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: [
          { user: 'What is the warranty?', answer: 'Two years.' },
          {
            user: 'Does it cover batteries?',
            answer: 'Batteries are covered for one year.',
            toolRounds: [
              {
                name: 'search_docs',
                args: { query: 'battery warranty' },
                result: 'Batteries: one year',
              },
            ],
          },
          { user: 'And accidental damage?', answer: 'Not covered.' },
          { user: 'Thanks.', answer: 'You are welcome.' },
        ].map(withRetrievedContext),
      },
      {
        conversationInstructions: CONVERSATION_B_CONTEXT,
        turns: [
          { user: 'Can I change my address?', answer: 'Yes, in settings.' },
          { user: 'Where in settings?', answer: 'Under Account.' },
        ].map(withRetrievedContext),
      },
    ],
  },
  {
    name: 'consecutive tool rounds with and without text',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: [
          {
            user: 'Check orders 1 and 2.',
            answer: 'Both ship today.',
            toolRounds: [
              { name: 'lookup_order', args: { query: '1' }, result: 'Order 1: ships today' },
              { name: 'lookup_order', args: { query: '2' }, result: 'Order 2: ships today' },
            ],
          },
          {
            user: 'Does the warranty cover them?',
            answer: 'Yes, two years for both.',
            toolRounds: [
              {
                text: 'Searching the documentation first.',
                name: 'search_docs',
                args: { query: 'warranty' },
                result: 'Warranty: two years',
              },
              { name: 'lookup_order', args: { query: '1' }, result: 'Order 1: covered' },
              {
                text: 'One more order to check.',
                name: 'lookup_order',
                args: { query: '2' },
                result: 'Order 2: covered',
              },
            ],
          },
          { user: 'Thanks.', answer: 'You are welcome.' },
        ],
      },
    ],
  },
  {
    name: 'conversation context changes mid-conversation',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: [
          { user: 'Hello.', answer: 'Hi.' },
          { user: 'What can you do?', answer: 'Answer support questions.' },
          {
            user: 'Remember that I prefer short answers.',
            answer: 'Noted.',
            conversationInstructions: `${CONVERSATION_A_CONTEXT}\n# Memory\nPrefers short answers.`,
          },
          { user: 'Where is order 7?', answer: 'In transit.' },
        ],
      },
    ],
  },
  {
    name: 'compaction inside a turn, tool rounds after it, next turns',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: IN_RUN_COMPACTION_TURNS,
      },
    ],
  },
  {
    name: 'compaction inside a turn beside per-turn volatile context',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: IN_RUN_COMPACTION_TURNS.map(withRetrievedContext),
      },
    ],
  },
  {
    name: 'host compaction appends a tool',
    instructions: STABLE_INSTRUCTIONS,
    tools: TOOLS,
    conversations: [
      {
        conversationInstructions: CONVERSATION_A_CONTEXT,
        turns: [
          { user: 'Start a refund for order 9.', answer: 'Started.' },
          { user: 'What is its status?', answer: 'Pending approval.' },
          {
            user: 'Escalate it.',
            answer: 'Escalated.',
            compaction: {
              summary:
                'The user asked to refund order 9; it is pending approval.',
              retainFromTurn: 1,
              appendTools: [
                { name: 'escalate_ticket', description: 'Escalates a ticket.' },
              ],
            },
          },
          { user: 'Any update?', answer: 'Not yet.' },
        ],
      },
    ],
  },
];

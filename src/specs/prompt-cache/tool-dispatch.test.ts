import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { CapturedRequest, Json, ToolSpec } from './harness';
import type { FakeToolCall, ProviderCase } from './providers';
import type * as t from '@/types';
import {
  toolSegments,
  stripCacheMarkers,
  commonPrefixLength,
  serializeInCacheOrder,
} from './harness';
import { TOOL_DISPATCHER_NAME } from '@/tools/toolDispatch';
import { createToolSearch } from '@/tools/ToolSearch';
import { formatAgentMessages } from '@/messages/format';
import { createContentAggregator } from '@/stream';
import { PROVIDER_CASES } from './providers';
import { Constants, GraphEvents } from '@/common';
import { Run } from '@/run';

/**
 * A tool made available mid-conversation through the dispatcher: the host
 * registers it as deferred and announces it, and the model reaches it through
 * `call_tool`. Model replies are faked in each provider's own streaming
 * format; the SDK runs the target, and the result lands in the next request
 * while `tools` never changes.
 */

jest.setTimeout(120_000);

const INSTRUCTIONS = 'You are a careful support assistant.';
const TOOLS: ToolSpec[] = [
  { name: 'lookup_order', description: 'Looks up an order by its identifier.' },
  { name: 'search_docs', description: 'Searches the product documentation.' },
];
const QUERY_SCHEMA: t.JsonSchemaType = {
  type: 'object',
  properties: { query: { type: 'string', description: 'Query' } },
  required: ['query'],
};
const TARGET: t.LCTool = {
  name: 'search_wiki',
  description: 'Searches the internal wiki.',
  parameters: QUERY_SCHEMA,
  defer_loading: true,
};
const ANNOUNCEMENT = `Tool \`${TARGET.name}\` is now available. Call it with \`${TOOL_DISPATCHER_NAME}\`.\n- ${TARGET.name}: ${TARGET.description}\n  parameters: ${JSON.stringify(QUERY_SCHEMA)}`;
const TARGET_ARGS = { query: 'bolts' };
const DISPATCHER_CALL: FakeToolCall = {
  id: 'call_dispatch_1',
  name: TOOL_DISPATCHER_NAME,
  args: { name: TARGET.name, arguments: JSON.stringify(TARGET_ARGS) },
};
const RESULT = 'wiki: bolts are in aisle 7';
const QUESTION = 'Where are the bolts?';

const REPLIES: FakeToolCall[] = [DISPATCHER_CALL];

const DECLARED_TOOLS = [
  ...TOOLS.map(({ name }) => name),
  Constants.TOOL_SEARCH,
  TOOL_DISPATCHER_NAME,
];

type Captured = {
  requests: CapturedRequest[];
  executed: Array<{ name: string; args: unknown }>;
  /** Content parts of the dispatching turn, as a host stores them. */
  stored: t.MessageContentComplex[];
};

function definitionOf({ name, description }: ToolSpec): t.LCTool {
  return { name, description, parameters: QUERY_SCHEMA };
}

function createAgent(
  providerCase: ProviderCase,
  targetAvailable: boolean
): t.AgentInputs {
  const definitions = [
    ...TOOLS.map(definitionOf),
    ...(targetAvailable ? [TARGET] : []),
  ];
  /** No registry at creation: its listing would change the tool description. */
  const toolSearch = createToolSearch({ mode: 'local' }) as t.GenericTool;
  return {
    agentId: 'prompt-cache-agent',
    provider: providerCase.provider,
    clientOptions: providerCase.clientOptions(),
    instructions: INSTRUCTIONS,
    maxContextTokens: 100_000,
    toolDispatch: true,
    toolRegistry: new Map(definitions.map((def) => [def.name, def])),
    toolDefinitions: definitions,
    graphTools: [toolSearch],
  } as t.AgentInputs;
}

function hostToolHandler(captured: Captured): t.EventHandler {
  return {
    handle: (_event: string, data: unknown): void => {
      const batch = data as t.ToolExecuteBatchRequest;
      batch.resolve(
        batch.toolCalls.map((call) => {
          captured.executed.push({ name: call.name, args: call.args });
          return { toolCallId: call.id, content: RESULT, status: 'success' };
        })
      );
    },
  };
}

function aggregatingHandlers(
  aggregate: t.ContentAggregator
): Record<string, t.EventHandler> {
  const forward = (event: string, data: unknown): void =>
    aggregate({ event, data } as Parameters<t.ContentAggregator>[0]);
  return Object.fromEntries(
    [
      GraphEvents.ON_RUN_STEP,
      GraphEvents.ON_RUN_STEP_DELTA,
      GraphEvents.ON_RUN_STEP_COMPLETED,
      GraphEvents.ON_MESSAGE_DELTA,
    ].map((event) => [event, { handle: forward }])
  );
}

/**
 * Runs one turn; `replies` answer the turn's first model requests with tool
 * calls, and the request after the last tool round is captured too. Returns
 * the turn's content parts as a host aggregates them.
 */
async function runTurn(
  providerCase: ProviderCase,
  agent: t.AgentInputs,
  captured: Captured,
  messages: BaseMessage[],
  replies: FakeToolCall[] = []
): Promise<t.MessageContentComplex[]> {
  const { contentParts, aggregateContent } = createContentAggregator();
  const pendingReplies = [...replies];
  const restore = providerCase.intercept(agent, (body: Json) => {
    captured.requests.push(providerCase.reduce(body));
    return pendingReplies.shift();
  });
  try {
    const runId = `tool-dispatch-${providerCase.name}-${Math.random()}`;
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: { type: 'standard', agents: [agent] },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        ...aggregatingHandlers(aggregateContent),
        [GraphEvents.ON_TOOL_EXECUTE]: hostToolHandler(captured),
      },
    });
    await run
      .processStream(
        { messages },
        { configurable: { thread_id: runId }, version: 'v2' }
      )
      .catch(() => undefined);
    return contentParts.filter(
      (part): part is t.MessageContentComplex => part != null
    );
  } finally {
    restore();
  }
}

async function runConversation(
  providerCase: ProviderCase
): Promise<Captured> {
  const captured: Captured = { requests: [], executed: [], stored: [] };
  const history: BaseMessage[] = [new HumanMessage('Hello.')];
  await runTurn(
    providerCase,
    createAgent(providerCase, false),
    captured,
    history
  );
  history.push(
    new AIMessage('Hi.'),
    new HumanMessage(`${ANNOUNCEMENT}\n\n${QUESTION}`)
  );
  captured.stored = await runTurn(
    providerCase,
    createAgent(providerCase, true),
    captured,
    history,
    REPLIES
  );
  /** The next turn's history is rebuilt from stored parts, as hosts do. */
  const { messages: restored } = formatAgentMessages(
    [
      {
        role: 'assistant',
        content: [...captured.stored, { type: 'text', text: 'In aisle 7.' }],
      },
    ],
    undefined,
    new Set(DECLARED_TOOLS)
  );
  history.push(...restored, new HumanMessage('Thanks.'));
  await runTurn(
    providerCase,
    createAgent(providerCase, true),
    captured,
    history
  );
  return captured;
}

const marker = (value: Json): string =>
  JSON.stringify(stripCacheMarkers(value));

function declaredToolName(segment: Json): string | undefined {
  if (segment == null || typeof segment !== 'object' || Array.isArray(segment)) {
    return undefined;
  }
  for (const holder of [segment, segment.function, segment.toolSpec]) {
    if (
      holder != null &&
      typeof holder === 'object' &&
      !Array.isArray(holder) &&
      typeof holder.name === 'string'
    ) {
      return holder.name;
    }
  }
  return undefined;
}

const CALL_FIELDS = ['arguments', 'input', 'args', 'response'];
const ARGUMENT_FIELDS = new Set(['input', 'args', 'arguments', 'response']);

/** Names of tool calls and results in every provider's message format. */
function calledToolNames(value: Json, names: string[] = []): string[] {
  if (Array.isArray(value)) {
    value.forEach((item) => calledToolNames(item, names));
    return names;
  }
  if (value == null || typeof value !== 'object') {
    return names;
  }
  if (
    typeof value.name === 'string' &&
    CALL_FIELDS.some((field) => field in value)
  ) {
    names.push(value.name);
  }
  for (const [key, nested] of Object.entries(value)) {
    if (!ARGUMENT_FIELDS.has(key)) {
      calledToolNames(nested, names);
    }
  }
  return names;
}

describe.each(PROVIDER_CASES.map((providerCase) => [providerCase.name, providerCase]))(
  'tool dispatch: %s',
  (_name, providerCase) => {
    let captured: Captured;
    /** The request after the dispatched tool round. */
    const afterDispatch = REPLIES.length + 1;

    beforeAll(async () => {
      captured = await runConversation(providerCase);
    });

    it('keeps tools byte-identical, with the dispatcher and without the target', () => {
      expect(captured.requests).toHaveLength(REPLIES.length + 3);
      const [first, ...rest] = captured.requests.map((request) =>
        toolSegments(request).map(marker)
      );
      for (const tools of rest) {
        expect(tools).toEqual(first);
      }
      const names = toolSegments(captured.requests[0]).map(declaredToolName);
      expect(names).toEqual(DECLARED_TOOLS);
    });

    it('starts every request with the previous one', () => {
      /** Responses restores a tool round as the provider's items, which is not part of this change. */
      const count =
        providerCase.name === 'openAI responses'
          ? captured.requests.length - 1
          : captured.requests.length;
      for (let i = 1; i < count; i++) {
        const before = serializeInCacheOrder(captured.requests[i - 1]);
        const after = serializeInCacheOrder(captured.requests[i]);
        expect(commonPrefixLength(before, after)).toBeGreaterThanOrEqual(
          before.length
        );
      }
    });

    it('answers the dispatcher call and names only declared tools in the history', () => {
      expect(captured.executed).toEqual([{ name: TARGET.name, args: TARGET_ARGS }]);
      for (const request of captured.requests.slice(afterDispatch)) {
        expect(JSON.stringify(request.messages)).toContain(RESULT);
      }
      for (const request of captured.requests) {
        const declared = new Set(toolSegments(request).map(declaredToolName));
        for (const name of calledToolNames(request.messages)) {
          expect(declared).toContain(name);
        }
      }
    });
  }
);

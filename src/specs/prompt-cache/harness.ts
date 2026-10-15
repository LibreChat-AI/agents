import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import type { ProviderCase } from './providers';
import { formatAgentMessages } from '@/messages';
import { ContentTypes } from '@/common';
import { Run } from '@/run';

/**
 * Offline prompt-cache harness: builds every turn of a scenario through a real
 * `Run`, captures the serialized provider request right before it would leave
 * the process, and reduces it to the order providers cache in (tools, system,
 * messages). See docs/prompt-cache-harness.md.
 */

export type Json =
  | string
  | number
  | boolean
  | null
  | Json[]
  | { [key: string]: Json };

export interface ToolSpec {
  name: string;
  description: string;
}

/** One model response that called a tool; consecutive rounds run sequentially. */
export interface ToolRound {
  name: string;
  args: Record<string, string>;
  result: string;
  /** Assistant text the model emitted before the call. */
  text?: string;
}

export interface TurnSpec {
  user: string;
  /** Final assistant text, appended to the history after this turn is captured. */
  answer: string;
  /** Tool rounds the assistant ran before `answer`; the follow-up call is captured too. */
  toolRounds?: ToolRound[];
  /** Per-turn volatile `additional_instructions`, for this turn only. */
  additionalInstructions?: string;
  /** Replaces `conversation_instructions` from this turn on (a declared reset point). */
  conversationInstructions?: string;
  /**
   * Host-side compaction before this turn (a declared reset point): the
   * summary becomes the run's initial summary, history restarts at
   * `retainFromTurn`, and `appendTools` join the end of the tool list.
   * With `afterRound`, the compaction happens inside this turn's run instead,
   * after that many tool rounds (the summarize node's path): the request that
   * follows is the reset point, and the next turns restore the summary and
   * the retained history as a host does. `appendTools` is not used then.
   */
  compaction?: {
    summary: string;
    retainFromTurn: number;
    appendTools?: ToolSpec[];
    afterRound?: number;
  };
}

export interface ConversationSpec {
  conversationInstructions: string;
  turns: TurnSpec[];
}

export interface Scenario {
  name: string;
  instructions: string;
  tools: ToolSpec[];
  conversations: ConversationSpec[];
}

/** Request reduced to cache order; marker-free copies are used for comparison. */
export interface CapturedRequest {
  tools: Json[];
  system: Json;
  messages: Json[];
  raw: Json;
}

export interface CapturedTurn {
  request: CapturedRequest;
  turn: TurnSpec;
  /** Tool list sent on this turn. */
  tools: ToolSpec[];
  /** Stable instructions shared by every conversation of the agent. */
  instructions: string;
  conversationInstructions: string;
  /** The model call after this turn's tool rounds, within the same turn. */
  followUp: boolean;
  /** First request after a declared change (compaction, new conversation context). */
  reset: boolean;
}

export const CAPTURED = 'prompt-cache harness: request captured before network';

function createTool({ name, description }: ToolSpec): t.GenericTool {
  return tool(async () => 'ok', {
    name,
    description,
    schema: {
      type: 'object',
      properties: { query: { type: 'string', description: 'Query' } },
      required: ['query'],
    },
  }) as unknown as t.GenericTool;
}

const callId = (turn: number, round: number): string =>
  `call_${turn}_${round}`;

/** Graph state after `count` rounds of the turn, as the live run holds it. */
function liveRoundMessages(
  turn: TurnSpec,
  index: number,
  count: number
): BaseMessage[] {
  return (turn.toolRounds ?? []).slice(0, count).flatMap((round, r) => [
    new AIMessage({
      content: round.text ?? '',
      tool_calls: [{ id: callId(index, r), name: round.name, args: round.args }],
    }),
    new ToolMessage({
      name: round.name,
      content: round.result,
      tool_call_id: callId(index, r),
    }),
  ]);
}

/** Content parts a host stores for a finished turn (as the aggregator builds them). */
function storedAnswerParts(turn: TurnSpec, index: number): t.MessageContentComplex[] {
  const parts = (turn.toolRounds ?? []).flatMap((round, r) => [
    ...(round.text != null
      ? [{ type: ContentTypes.TEXT, text: round.text } as t.MessageContentComplex]
      : []),
    {
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: callId(index, r),
        name: round.name,
        args: JSON.stringify(round.args),
        output: round.result,
        progress: 1,
        roundId: `step_${index}_${r}`,
      },
    } as t.MessageContentComplex,
  ]);
  return [...parts, { type: ContentTypes.TEXT, text: turn.answer }];
}

/** History restored from stored parts the way a host formats it each turn. */
function buildHistory(
  turns: TurnSpec[],
  current: number,
  retainFromTurn: number
): BaseMessage[] {
  const payload: t.TPayload = [];
  for (let i = retainFromTurn; i <= current; i++) {
    payload.push({ role: 'user', content: turns[i].user });
    if (i < current) {
      payload.push({ role: 'assistant', content: storedAnswerParts(turns[i], i) });
    }
  }
  const tools = new Set(turns.flatMap((turn) => (turn.toolRounds ?? []).map((round) => round.name)));
  return formatAgentMessages(payload, undefined, tools).messages;
}

async function captureRequest(
  providerCase: ProviderCase,
  agent: t.AgentInputs,
  messages: BaseMessage[],
  runSummary?: string
): Promise<CapturedRequest> {
  let captured: CapturedRequest | undefined;
  const record = (body: Json): void => {
    captured = providerCase.reduce(body);
  };
  const restore = providerCase.intercept(agent, record);
  try {
    const runId = `prompt-cache-${providerCase.name}-${Math.random()}`;
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: { type: 'standard', agents: [agent] },
      returnContent: true,
      skipCleanup: true,
    });
    if (runSummary != null) {
      // The state the summarize node leaves for the rest of its run.
      run.Graph?.agentContexts.get(agent.agentId)?.setSummary(runSummary, 20);
    }
    await run
      .processStream(
        { messages },
        { configurable: { thread_id: runId }, version: 'v2' }
      )
      .catch(() => undefined);
  } finally {
    restore();
  }
  if (captured == null) {
    throw new Error(`No request captured for ${providerCase.name}`);
  }
  return captured;
}

/** Runs one conversation of a scenario and captures every turn. */
export async function runConversation(
  providerCase: ProviderCase,
  scenario: Scenario,
  conversation: ConversationSpec
): Promise<CapturedTurn[]> {
  const captured: CapturedTurn[] = [];
  let tools = scenario.tools;
  let conversationInstructions = conversation.conversationInstructions;
  let summary: string | undefined;
  let retainFromTurn = 0;

  for (let i = 0; i < conversation.turns.length; i++) {
    const turn = conversation.turns[i];
    if (turn.conversationInstructions != null) {
      conversationInstructions = turn.conversationInstructions;
    }
    const { compaction } = turn;
    const inRunAfter = compaction?.afterRound;
    if (compaction != null && inRunAfter == null) {
      summary = compaction.summary;
      retainFromTurn = compaction.retainFromTurn;
      tools = [...tools, ...(compaction.appendTools ?? [])];
    }
    const agent: t.AgentInputs = {
      agentId: 'prompt-cache-agent',
      provider: providerCase.provider,
      clientOptions: providerCase.clientOptions(),
      instructions: scenario.instructions,
      conversation_instructions: conversationInstructions,
      additional_instructions: turn.additionalInstructions,
      maxContextTokens: 100_000,
      ...(summary != null
        ? { initialSummary: { text: summary, tokenCount: 20 } }
        : {}),
    } as t.AgentInputs;
    const rounds = turn.toolRounds?.length ?? 0;
    for (let call = 0; call <= rounds; call++) {
      const compacted = inRunAfter != null && call >= inRunAfter;
      const history = buildHistory(
        conversation.turns,
        i,
        compacted ? compaction!.retainFromTurn : retainFromTurn
      );
      const request = await captureRequest(
        providerCase,
        { ...agent, tools: tools.map(createTool) } as t.AgentInputs,
        [...history, ...liveRoundMessages(turn, i, call)],
        compacted ? compaction!.summary : undefined
      );
      captured.push({
        request,
        turn,
        tools,
        instructions: scenario.instructions,
        conversationInstructions,
        followUp: call > 0,
        reset: compacted
          ? call === inRunAfter
          : call === 0 &&
            ((compaction != null && inRunAfter == null) ||
              turn.conversationInstructions != null),
      });
    }
    if (compaction != null && inRunAfter != null) {
      summary = compaction.summary;
      retainFromTurn = compaction.retainFromTurn;
    }
  }
  return captured;
}

function isCachePoint(value: Json): boolean {
  return (
    value != null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.keys(value).length === 1 &&
    'cachePoint' in value
  );
}

/**
 * Cache markers are not prompt content; they move to the newest turn by design.
 * Marking also turns string content into a single text block, which providers
 * read as the same prompt, so string content is compared in block form.
 */
export function stripCacheMarkers(value: Json): Json {
  if (Array.isArray(value)) {
    return value
      .filter((item) => !isCachePoint(item))
      .map((item) => stripCacheMarkers(item));
  }
  if (value != null && typeof value === 'object') {
    const entries: [string, Json][] = [];
    for (const key of Object.keys(value)) {
      const nested = value[key];
      if (key === 'cache_control') {
        continue;
      }
      entries.push([
        key,
        key === 'content' && typeof nested === 'string'
          ? [{ type: 'text', text: nested }]
          : stripCacheMarkers(nested),
      ]);
    }
    return Object.fromEntries(entries);
  }
  return value;
}

/** Tools, system, then messages, one marker-free JSON segment per line. */
export function serializeInCacheOrder(request: CapturedRequest): string {
  return [...toolSegments(request), request.system, ...request.messages]
    .map((segment) => JSON.stringify(stripCacheMarkers(segment)))
    .join('\n');
}

/** Tool entries without standalone cache-point entries (Bedrock `toolConfig`). */
export function toolSegments(request: CapturedRequest): Json[] {
  return request.tools.filter((segment) => !isCachePoint(segment));
}

export function commonPrefixLength(a: string, b: string): number {
  const limit = Math.min(a.length, b.length);
  let i = 0;
  while (i < limit && a.charCodeAt(i) === b.charCodeAt(i)) {
    i++;
  }
  return i;
}

/** Offset in the serialized request right after the last tool segment. */
export function endOfTools(request: CapturedRequest): number {
  return toolSegments(request)
    .map((segment) => JSON.stringify(stripCacheMarkers(segment)).length + 1)
    .reduce((sum, length) => sum + length, 0);
}

/** Offset right after the first occurrence of `text` (JSON-escaped). */
export function endOfText(serialized: string, text: string): number {
  const needle = JSON.stringify(text).slice(1, -1);
  const index = serialized.indexOf(needle);
  return index < 0 ? -1 : index + needle.length;
}

export function countMatches(value: Json, pattern: RegExp): number {
  return JSON.stringify(value).match(pattern)?.length ?? 0;
}

import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import type { Json } from './harness';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { GraphEvents } from '@/common';
import { formatAgentMessages } from '@/messages';
import { CAPTURED } from './harness';
import { createTokenCounter, encodingForModel } from '@/utils/tokens';
import { Run } from '@/run';

/**
 * Drives real runs against scripted provider streams (fetch is replaced, no
 * network) and restores each finished turn from the content parts the SDK
 * aggregator built, the way a host persists and formats history.
 */

export type JsonObject = { [key: string]: Json };

export interface LiveTurn {
  /** Request bodies sent during this turn, in order. */
  requests: JsonObject[];
  contentParts: t.MessageContentComplex[];
}

export function sseResponse(frames: string[]): Response {
  return new Response(frames.join(''), {
    headers: { 'content-type': 'text/event-stream' },
  });
}

/** Tool output as a host returns it: plain text or content blocks. */
export type ToolOutput = string | t.MessageContentComplex[];

/** Host-side tool executor for event-driven tools (`toolDefinitions`). */
function createToolExecuteHandler(
  results: Map<string, ToolOutput>
): t.EventHandler {
  return {
    handle: (_event, data): void => {
      const request = data as unknown as t.ToolExecuteBatchRequest;
      request.resolve(
        request.toolCalls.map((call) => ({
          toolCallId: call.id,
          content: results.get(call.name) ?? 'ok',
          status: 'success' as const,
        }))
      );
    },
  };
}

function aggregatingHandlers(
  aggregateContent: t.ContentAggregator,
  events: GraphEvents[]
): Record<string, t.EventHandler> {
  return Object.fromEntries(
    events.map((event) => [
      event,
      {
        handle: (_event: string, data: t.StreamEventData): void =>
          aggregateContent({ event, data } as Parameters<
            t.ContentAggregator
          >[0]),
      },
    ])
  );
}

export const AGGREGATED_EVENTS = [
  GraphEvents.ON_RUN_STEP,
  GraphEvents.ON_RUN_STEP_DELTA,
  GraphEvents.ON_RUN_STEP_COMPLETED,
  GraphEvents.ON_MESSAGE_DELTA,
  GraphEvents.ON_REASONING_DELTA,
];

export interface LiveRunOptions {
  agent: t.AgentInputs;
  /** Tool name -> result; tools run event-driven when `eventDriven` is set. */
  toolResults: Map<string, ToolOutput>;
  eventDriven: boolean;
  /** Extra events the host routes to the aggregator. */
  aggregatedEvents?: GraphEvents[];
  /** Bedrock takes an injected client instead of `fetch`. */
  transport?: 'fetch' | 'bedrock';
  /** Turns the context management on: pruning, fading and masking need counts. */
  contextManagement?: boolean;
}

type Restore = () => void;

/** A scripted reply that fails the call, to exercise fallbacks. */
export interface ProviderFailure {
  failWithStatus: number;
  /** Error text, for providers whose errors carry one (Bedrock). */
  message?: string;
}

/** Serves scripted replies in order; a call past the script is recorded and aborted. */
function installTransport(
  options: LiveRunOptions,
  requests: JsonObject[],
  replies: unknown[]
): Restore {
  const queue = [...replies];
  if (options.transport === 'bedrock') {
    const clientOptions = options.agent.clientOptions as { client?: object };
    clientOptions.client = {
      send: async (command: { input: object }): Promise<object> => {
        requests.push(JSON.parse(JSON.stringify(command.input)) as JsonObject);
        const events = queue.shift() as object[] | ProviderFailure | undefined;
        if (events == null) {
          throw new Error(CAPTURED);
        }
        if (!Array.isArray(events)) {
          throw Object.assign(new Error(events.message ?? 'scripted failure'), {
            name: 'ValidationException',
            $metadata: { httpStatusCode: events.failWithStatus },
          });
        }
        return {
          stream: (async function* (): AsyncGenerator<object> {
            for (const event of events) {
              yield event;
            }
          })(),
        };
      },
      config: {},
      destroy: (): void => undefined,
      middlewareStack: { add: (): void => undefined, use: (): void => undefined },
    };
    return () => {
      delete clientOptions.client;
    };
  }
  const spy = jest
    .spyOn(global, 'fetch')
    .mockImplementation(async (input, init) => {
      const raw =
        input instanceof Request
          ? await input.clone().text()
          : String(init?.body ?? '{}');
      requests.push(JSON.parse(raw) as JsonObject);
      const frames = queue.shift() as string[] | ProviderFailure | undefined;
      if (frames == null) {
        throw new Error(CAPTURED);
      }
      if (!Array.isArray(frames)) {
        return new Response(
          JSON.stringify({
            type: 'error',
            error: { type: 'api_error', message: 'scripted failure' },
          }),
          { status: frames.failWithStatus }
        );
      }
      return sseResponse(frames);
    });
  return () => spy.mockRestore();
}

/** Runs one turn; `replies` answer the model calls in order, a further call is captured and aborted. */
export async function runLiveTurn(
  options: LiveRunOptions,
  messages: BaseMessage[],
  replies: unknown[]
): Promise<LiveTurn> {
  const requests: JsonObject[] = [];
  const restore = installTransport(options, requests, replies);
  const { contentParts, aggregateContent } = createContentAggregator();
  try {
    const runId = `live-${Math.random()}`;
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: { type: 'standard', agents: [options.agent] },
      returnContent: true,
      skipCleanup: true,
      ...(options.contextManagement === true
        ? { tokenCounter: await createTokenCounter(encodingForModel('gpt-4')), indexTokenCountMap: {} }
        : {}),
      ...(options.eventDriven
        ? { eagerEventToolExecution: { enabled: true } }
        : {}),
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_TOOL_EXECUTE]: createToolExecuteHandler(
          options.toolResults
        ),
        ...aggregatingHandlers(aggregateContent, [
          ...AGGREGATED_EVENTS,
          ...(options.aggregatedEvents ?? []),
        ]),
      },
    });
    await run
      .processStream(
        { messages },
        { configurable: { thread_id: runId }, version: 'v2' }
      )
      .catch(() => undefined);
  } finally {
    restore();
  }
  return {
    requests,
    contentParts: contentParts.filter(
      (part): part is t.MessageContentComplex => part != null
    ),
  };
}

/** History a host restores before the next turn from the stored parts. */
export function restoreHistory(
  exchanges: Array<{ user: string; parts?: t.MessageContentComplex[] }>,
  toolNames: Set<string>,
  replayProviderReasoning?: { provider: t.ProviderName; model: string }
): BaseMessage[] {
  const payload: t.TPayload = exchanges.flatMap(({ user, parts }) => [
    { role: 'user', content: user },
    ...(parts != null ? [{ role: 'assistant', content: parts }] : []),
  ]);
  return formatAgentMessages(
    payload,
    undefined,
    toolNames,
    undefined,
    replayProviderReasoning == null ? undefined : { replayProviderReasoning }
  ).messages;
}

export function firstTurnMessages(user: string): BaseMessage[] {
  return restoreHistory([{ user }], new Set());
}

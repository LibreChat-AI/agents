import { HumanMessage, isAIMessage } from '@langchain/core/messages';
import type {
  Content,
  EnhancedGenerateContentResponse,
  GenerateContentRequest,
  Part,
} from '@google/generative-ai';
import type { BaseMessage } from '@langchain/core/messages';
import type { GeminiPart } from '@langchain/google-common';
import type * as t from '@/types';
import type { TPayload } from '@/types';
import {
  GEMINI_PARTS,
  isGeminiReplayParts,
  restoreGeminiReplayParts,
} from '@/llm/google/utils/signatures';
import type { GeminiReplayPart } from '@/llm/google/utils/signatures';
import { GoogleTransport } from '@/llm/google/transport';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { formatAgentMessages } from './format';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

describe('Gemini 3 tool rounds restore as sent live', () => {
  const lookupCall = (city: string, thoughtSignature?: string): Part =>
    ({
      functionCall: { name: 'lookup', args: { city } },
      ...(thoughtSignature != null && { thoughtSignature }),
    }) as Part;
  const thought = (text: string): Part => ({ text, thought: true }) as Part;
  /** One model response per round, streamed part by part. */
  const rounds: Part[][] = [
    [
      thought('Plan one.'),
      { text: 'Checking Rome.' },
      lookupCall('Rome', 'sig-1'),
    ],
    [thought('Plan two.'), lookupCall('Paris', 'sig-2')],
    [
      thought('Plan three.'),
      { text: 'Two more.' },
      lookupCall('Oslo', 'sig-3'),
      lookupCall('Lima'),
    ],
    [
      thought('Plan four.'),
      { text: 'All sunny.', thoughtSignature: 'sig-4' } as Part,
    ],
  ];
  const storedSignatures = ['sig-1', 'sig-2', 'sig-3', 'sig-4'];

  const response = (
    part: Part,
    last: boolean
  ): EnhancedGenerateContentResponse =>
    ({
      candidates: [
        {
          content: { role: 'model', parts: [part] },
          index: 0,
          ...(last && { finishReason: 'STOP' }),
        },
      ],
    }) as EnhancedGenerateContentResponse;

  /**
   * Runs one turn the way LibreChat does: event-driven tools with eager
   * execution, the content aggregator as the stored message, and the native
   * parts of every signed model call as `metadata.thoughtSignatures.turns`.
   */
  async function runTurn(
    messages: BaseMessage[],
    responses: Part[][]
  ): Promise<{
    requests: GenerateContentRequest[];
    content: TPayload[number]['content'];
    turns: GeminiReplayPart[][];
  }> {
    const requests: GenerateContentRequest[] = [];
    jest
      .spyOn(GoogleTransport.prototype, 'generateContentStream')
      .mockImplementation(async (request) => {
        requests.push(JSON.parse(JSON.stringify(request)));
        const parts = responses[requests.length - 1];
        return {
          stream: (async function* () {
            for (let i = 0; i < parts.length; i++) {
              yield response(parts[i], i === parts.length - 1);
            }
          })(),
        };
      });
    const { contentParts, aggregateContent } = createContentAggregator();
    const aggregate: t.EventHandler = {
      handle: (event, data) =>
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.RunStep,
        }),
    };
    const turns: GeminiReplayPart[][] = [];
    const threadId = `gemini-rounds-${messages.length}`;
    const run = await Run.create<t.IState>({
      runId: threadId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'gemini',
            provider: Providers.GOOGLE,
            clientOptions: {
              model: 'gemini-3-pro-preview',
              apiKey: 'test-key',
              streaming: true,
              streamUsage: false,
              _lc_stream_delay: 0,
            } as t.ClientOptions,
            reasoningKey: 'reasoning',
            toolDefinitions: [
              {
                name: 'lookup',
                description: 'Weather lookup',
                parameters: {
                  type: 'object',
                  properties: { city: { type: 'string' } },
                  required: ['city'],
                },
              },
            ],
          },
        ],
      },
      eagerEventToolExecution: { enabled: true },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_RUN_STEP]: aggregate,
        [GraphEvents.ON_RUN_STEP_DELTA]: aggregate,
        [GraphEvents.ON_RUN_STEP_COMPLETED]: aggregate,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregate,
        [GraphEvents.ON_REASONING_DELTA]: aggregate,
        [GraphEvents.ON_TOOL_EXECUTE]: {
          handle: (_event, data): void => {
            const batch = data as t.ToolExecuteBatchRequest;
            batch.resolve(
              batch.toolCalls.map((call) => ({
                toolCallId: call.id,
                status: 'success' as const,
                content: `Sunny in ${(call.args as { city: string }).city}`,
              }))
            );
          },
        },
        [GraphEvents.CHAT_MODEL_END]: {
          handle: (_event, data): void => {
            const parts = (data as t.ModelEndData)?.output?.additional_kwargs[
              GEMINI_PARTS
            ];
            if (
              isGeminiReplayParts(parts) &&
              parts.some((part) => part.thoughtSignature != null)
            ) {
              turns.push(parts);
            }
          },
        },
      },
    });
    await run.processStream(
      { messages },
      { configurable: { thread_id: threadId }, version: 'v2' }
    );
    return {
      requests,
      content: JSON.parse(JSON.stringify(contentParts)),
      turns,
    };
  }

  /** LibreChat `restoreGoogleThoughtSignatures`: a message takes the one stored call it equals. */
  function restoreTurns(
    messages: BaseMessage[],
    turns: GeminiReplayPart[][]
  ): void {
    for (const message of messages) {
      if (!isAIMessage(message)) continue;
      const matches = turns.filter(
        (turn) => restoreGeminiReplayParts(message, turn) != null
      );
      if (matches.length === 1) {
        message.additional_kwargs[GEMINI_PARTS] = matches[0];
      }
    }
  }

  afterEach(() => jest.restoreAllMocks());

  it('replays every round as its own model turn with its thoughts and signatures', async () => {
    const live = await runTurn(
      [new HumanMessage('Weather in four cities?')],
      rounds
    );
    expect(live.requests).toHaveLength(rounds.length);
    expect(live.turns).toHaveLength(rounds.length);

    const { messages } = formatAgentMessages(
      [
        { role: 'user', content: 'Weather in four cities?' },
        { role: 'assistant', messageId: 'answer', content: live.content },
      ],
      undefined,
      undefined,
      undefined,
      { preserveTextContent: true, preserveReasoningContent: true }
    );
    restoreTurns(messages, live.turns);
    const next = await runTurn(
      [...messages, new HumanMessage('Sure?')],
      [[{ text: 'Yes.', thoughtSignature: 'sig-5' } as Part]]
    );

    const history = next.requests[0].contents;
    const modelTurns = history
      .filter((content) => content.role === 'model')
      .map((content) => content.parts as GeminiPart[]);
    const signatures = modelTurns.flatMap((parts) =>
      parts.flatMap((part) =>
        part.thoughtSignature != null ? [part.thoughtSignature] : []
      )
    );
    expect(modelTurns).toHaveLength(rounds.length);
    expect(
      modelTurns.map((parts) =>
        parts.flatMap((part) =>
          part.thought === true && 'text' in part ? [part.text] : []
        )
      )
    ).toEqual([['Plan one.'], ['Plan two.'], ['Plan three.'], ['Plan four.']]);
    expect(signatures).toEqual(storedSignatures);
    const expected: Content[] = [
      ...live.requests[rounds.length - 1].contents,
      { role: 'model', parts: rounds[rounds.length - 1] },
      { role: 'user', parts: [{ text: 'Sure?' }] },
    ];
    expect(JSON.stringify(history)).toBe(JSON.stringify(expected));
  });
});

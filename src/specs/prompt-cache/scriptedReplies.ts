import type { Json } from './harness';
import type { JsonObject } from './liveRun';

/**
 * Provider-neutral model replies rendered into each wire protocol's stream,
 * so a real run can be driven offline through every provider client.
 */

export type Protocol = 'chat' | 'responses' | 'anthropic' | 'google' | 'bedrock';

export interface ScriptedCall {
  name: string;
  args: Record<string, string>;
}

/** Signed thinking, or Anthropic `redacted_thinking` with opaque `data`. */
export type ScriptedReasoning =
  | { text: string; signature: string }
  | { redacted: string };

export interface ScriptedReply {
  /** Reasoning blocks before the output, in order. */
  reasoning?: ScriptedReasoning[];
  /** Reasoning emitted after the text and before the calls (interleaved thinking). */
  reasoningAfterText?: ScriptedReasoning[];
  text?: string;
  calls?: ScriptedCall[];
}

let replyCount = 0;

const sse = (event: JsonObject, name?: string): string =>
  `${name != null ? `event: ${name}\n` : ''}data: ${JSON.stringify(event)}\n\n`;

function chatStream(reply: ScriptedReply, n: number): string[] {
  const chunk = (delta: JsonObject, finish: string | null = null): string =>
    sse({
      id: `chatcmpl-${n}`,
      object: 'chat.completion.chunk',
      created: 1,
      model: 'scripted',
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  const calls = reply.calls ?? [];
  return [
    chunk({ role: 'assistant', content: reply.text ?? null }),
    ...calls.map((call, index) =>
      chunk({
        tool_calls: [
          {
            index,
            id: `call_${n}_${index}`,
            type: 'function',
            function: { name: call.name, arguments: JSON.stringify(call.args) },
          },
        ],
      })
    ),
    chunk({}, calls.length > 0 ? 'tool_calls' : 'stop'),
    'data: [DONE]\n\n',
  ];
}

function responsesStream(reply: ScriptedReply, n: number): string[] {
  const output: JsonObject[] = [];
  if (reply.text != null) {
    output.push({
      id: `msg_${n}`,
      type: 'message',
      status: 'completed',
      content: [{ type: 'output_text', annotations: [], logprobs: [], text: reply.text }],
      role: 'assistant',
    });
  }
  (reply.calls ?? []).forEach((call, index) =>
    output.push({
      id: `fc_${n}_${index}`,
      type: 'function_call',
      status: 'completed',
      arguments: JSON.stringify(call.args),
      call_id: `call_${n}_${index}`,
      name: call.name,
    })
  );
  const response = {
    id: `resp_${n}`,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model: 'scripted',
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens_details: { reasoning_tokens: 0 },
    },
  };
  const frames = [
    sse({ type: 'response.created', sequence_number: 0, response: { ...response, status: 'in_progress', output: [] } }, 'response.created'),
  ];
  output.forEach((item, outputIndex) => {
    const isMessage = item.type === 'message';
    frames.push(
      sse(
        {
          type: 'response.output_item.added',
          sequence_number: frames.length,
          output_index: outputIndex,
          item: isMessage
            ? { ...item, status: 'in_progress', content: [] }
            : { ...item, status: 'in_progress', arguments: '' },
        },
        'response.output_item.added'
      ),
      isMessage
        ? sse(
          {
            type: 'response.output_text.delta',
            sequence_number: frames.length + 1,
            item_id: item.id,
            output_index: outputIndex,
            content_index: 0,
            delta: reply.text ?? '',
            logprobs: [],
          },
          'response.output_text.delta'
        )
        : sse(
          {
            type: 'response.function_call_arguments.delta',
            sequence_number: frames.length + 1,
            item_id: item.id,
            output_index: outputIndex,
            delta: item.arguments,
          },
          'response.function_call_arguments.delta'
        ),
      sse(
        { type: 'response.output_item.done', sequence_number: frames.length + 2, output_index: outputIndex, item },
        'response.output_item.done'
      )
    );
  });
  frames.push(
    sse(
      { type: 'response.completed', sequence_number: frames.length, response: { ...response, status: 'completed', output } },
      'response.completed'
    )
  );
  return frames;
}

function anthropicStream(reply: ScriptedReply, n: number): string[] {
  const event = (body: JsonObject): string => sse(body, String(body.type));
  const blocks: string[][] = [];
  const pushReasoning = (list: ScriptedReasoning[] | undefined): void => {
    for (const reasoning of list ?? []) {
      blocks.push(
        'redacted' in reasoning
          ? [JSON.stringify({ type: 'redacted_thinking', data: reasoning.redacted })]
          : [
            JSON.stringify({ type: 'thinking', thinking: '', signature: '' }),
            JSON.stringify({ type: 'thinking_delta', thinking: reasoning.text }),
            JSON.stringify({ type: 'signature_delta', signature: reasoning.signature }),
          ]
      );
    }
  };
  pushReasoning(reply.reasoning);
  if (reply.text != null) {
    blocks.push([
      JSON.stringify({ type: 'text', text: '' }),
      JSON.stringify({ type: 'text_delta', text: reply.text }),
    ]);
  }
  pushReasoning(reply.reasoningAfterText);
  (reply.calls ?? []).forEach((call, index) =>
    blocks.push([
      JSON.stringify({ type: 'tool_use', id: `toolu_${n}_${index}`, name: call.name, input: {} }),
      JSON.stringify({ type: 'input_json_delta', partial_json: JSON.stringify(call.args) }),
    ])
  );
  const hasCalls = (reply.calls?.length ?? 0) > 0;
  return [
    event({
      type: 'message_start',
      message: {
        id: `msg_${n}`,
        type: 'message',
        role: 'assistant',
        model: 'scripted',
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 1 },
      },
    }),
    ...blocks.flatMap(([start, ...deltas], index) => [
      event({ type: 'content_block_start', index, content_block: JSON.parse(start) as Json }),
      ...deltas.map((delta) =>
        event({ type: 'content_block_delta', index, delta: JSON.parse(delta) as Json })
      ),
      event({ type: 'content_block_stop', index }),
    ]),
    event({
      type: 'message_delta',
      delta: { stop_reason: hasCalls ? 'tool_use' : 'end_turn', stop_sequence: null },
      usage: { output_tokens: 5 },
    }),
    event({ type: 'message_stop' }),
  ];
}

function googleStream(reply: ScriptedReply): string[] {
  const parts: JsonObject[] = [
    ...(reply.text != null ? [{ text: reply.text }] : []),
    ...(reply.calls ?? []).map((call) => ({
      functionCall: { name: call.name, args: call.args },
    })),
  ];
  return parts.map((part, index) =>
    sse({
      candidates: [
        {
          content: { role: 'model', parts: [part] },
          ...(index === parts.length - 1 && { finishReason: 'STOP' }),
          index: 0,
        },
      ],
      ...(index === parts.length - 1 && {
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
      }),
      modelVersion: 'scripted',
    })
  );
}

function bedrockEvents(reply: ScriptedReply, n: number): JsonObject[] {
  const events: JsonObject[] = [{ messageStart: { role: 'assistant' } }];
  let index = 0;
  const pushReasoning = (list: ScriptedReasoning[] | undefined): void => {
    for (const reasoning of list ?? []) {
      if ('redacted' in reasoning) {
        continue;
      }
      events.push(
        { contentBlockDelta: { contentBlockIndex: index, delta: { reasoningContent: { text: reasoning.text } } } },
        { contentBlockDelta: { contentBlockIndex: index, delta: { reasoningContent: { signature: reasoning.signature } } } },
        { contentBlockStop: { contentBlockIndex: index } }
      );
      index++;
    }
  };
  pushReasoning(reply.reasoning);
  if (reply.text != null) {
    events.push(
      { contentBlockDelta: { contentBlockIndex: index, delta: { text: reply.text } } },
      { contentBlockStop: { contentBlockIndex: index } }
    );
    index++;
  }
  pushReasoning(reply.reasoningAfterText);
  for (const [callIndex, call] of (reply.calls ?? []).entries()) {
    events.push(
      {
        contentBlockStart: {
          contentBlockIndex: index,
          start: { toolUse: { toolUseId: `tooluse_${n}_${callIndex}`, name: call.name } },
        },
      },
      { contentBlockDelta: { contentBlockIndex: index, delta: { toolUse: { input: JSON.stringify(call.args) } } } },
      { contentBlockStop: { contentBlockIndex: index } }
    );
    index++;
  }
  events.push(
    { messageStop: { stopReason: (reply.calls?.length ?? 0) > 0 ? 'tool_use' : 'end_turn' } },
    { metadata: { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 }, metrics: { latencyMs: 1 } } }
  );
  return events;
}

/** One model reply in the protocol's wire form: SSE frames, or Bedrock stream events. */
export function scriptReply(protocol: Protocol, reply: ScriptedReply): unknown {
  const n = ++replyCount;
  switch (protocol) {
  case 'chat':
    return chatStream(reply, n);
  case 'responses':
    return responsesStream(reply, n);
  case 'anthropic':
    return anthropicStream(reply, n);
  case 'google':
    return googleStream(reply);
  case 'bedrock':
    return bedrockEvents(reply, n);
  }
}

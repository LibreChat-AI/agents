import { MemorySaver, Send } from '@langchain/langgraph';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { ToolResultProtection } from '@/protection/toolResult';
import type { SubagentGraphResult } from '@/types';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { FakeChatModel } from '@/llm/fake';
import { StandardGraph } from '../Graph';
import { Providers } from '@/common';
import { Run } from '@/run';

const rawResult = 'selected-result@example.invalid';
const metadataFields = [
  'additional_kwargs',
  'response_metadata',
  'lc_kwargs',
] as const;
const variants = [
  { name: 'AIMessage', Message: AIMessage },
  { name: 'AIMessageChunk', Message: AIMessageChunk },
];

const makeGraph = (): StandardGraph =>
  new StandardGraph({
    runId: 'result-metadata-protection',
    agents: [{ agentId: 'result', provider: Providers.OPENAI }],
  });

const makeMessages = (
  Message: typeof AIMessage | typeof AIMessageChunk,
  id?: string
) => {
  const fields = () => ({
    id,
    name: 'result-agent',
    content: [{ type: 'text' as const, text: 'Allowed designated answer' }],
    tool_calls: [
      {
        id: 'allowed-call',
        name: 'lookup',
        args: { query: 'Allowed argument', options: { count: 42 } },
        type: 'tool_call' as const,
      },
    ],
    additional_kwargs: { allowed: 'provider-detail' },
    response_metadata: { finish_reason: 'tool_calls' },
    usage_metadata: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
  });
  const canonical = new Message(fields());
  const supplied = new Message(fields());
  canonical.lc_kwargs.allowed = 'serialization-detail';
  supplied.lc_kwargs.allowed = 'serialization-detail';
  return {
    canonical,
    supplied,
    input: { agentId: 'result', message: supplied },
  };
};

const expectCanonical = (
  result: SubagentGraphResult,
  canonical: AIMessage | AIMessageChunk
): void => {
  expect(result.agentId).toBe('result');
  expect(result.message).toBeDefined();
  const message = result.message as AIMessage | AIMessageChunk;
  expect(Object.getPrototypeOf(message)).toBe(Object.getPrototypeOf(canonical));
  expect(message.id).toBe(canonical.id);
  expect(message.name).toBe(canonical.name);
  expect(message.content).toEqual(canonical.content);
  expect(message.tool_calls).toEqual(canonical.tool_calls);
  expect(message.usage_metadata).toEqual(canonical.usage_metadata);
  expect(message.additional_kwargs).toEqual(canonical.additional_kwargs);
  expect(message.response_metadata).toEqual(canonical.response_metadata);
  expect(message.lc_kwargs).toEqual(canonical.lc_kwargs);
  expect(JSON.stringify(result)).not.toContain(rawResult);
  expect(JSON.stringify(message.toJSON())).not.toContain(rawResult);
};

describe.each(variants)('$name result metadata protection', ({ Message }) => {
  it.each(metadataFields)(
    'replaces supplied %s with canonical history metadata',
    (field) => {
      const { canonical, supplied, input } = makeMessages(
        Message,
        'designated-result'
      );
      supplied[field] = { selectedResult: rawResult };

      const result = makeGraph().validateToolResultSubagentResult(
        input,
        [canonical],
        'result'
      );

      expectCanonical(result, canonical);
      expect(result).not.toBe(input);
      expect(result.message).not.toBe(supplied);
      expect(result.message).not.toBe(canonical);
    }
  );

  it.each(['designated-result', undefined])(
    'preserves allowed metadata and arguments with id %s',
    (id) => {
      const { canonical, input } = makeMessages(Message, id);
      const result = makeGraph().validateToolResultSubagentResult(input, [
        canonical,
      ]);

      expectCanonical(result, canonical);
      expect(result.message).not.toBe(canonical);
      expect((result.message as AIMessage).tool_calls?.[0].args).toBe(
        canonical.tool_calls?.[0].args
      );
      expect(result.message!.lc_kwargs.tool_calls).toBe(
        (result.message as AIMessage).tool_calls
      );
    }
  );

  it.each(metadataFields)('detaches late nested %s aliases', (field) => {
    const { canonical, input } = makeMessages(Message, 'designated-result');
    const retained = { items: [{ value: 'Allowed nested metadata' }] };
    canonical[field].retained = retained;
    const result = makeGraph().validateToolResultSubagentResult(input, [
      canonical,
    ]);
    const before = JSON.stringify(result);
    const message = result.message as AIMessage | AIMessageChunk;
    expect(message[field].retained).toEqual(retained);
    expect(message[field].retained).not.toBe(retained);

    retained.items[0].value = rawResult;
    retained.items.push({ value: rawResult });

    expect(JSON.stringify(result)).toBe(before);
    expect(JSON.stringify(message.toJSON())).not.toContain(rawResult);
    expect((message as AIMessage).tool_calls?.[0].args).toBe(
      canonical.tool_calls?.[0].args
    );
  });

  it.each(['additional_kwargs', 'response_metadata'] as const)(
    'synchronizes canonical %s into lc_kwargs instead of stale serializer metadata',
    (field) => {
      const { canonical, input } = makeMessages(Message, 'designated-result');
      canonical.lc_kwargs[field] = { selectedResult: rawResult };
      const result = makeGraph().validateToolResultSubagentResult(input, [
        canonical,
      ]);
      const message = result.message!;

      expect(message[field]).toEqual(canonical[field]);
      expect(message.lc_kwargs[field]).toEqual(message[field]);
      expect(JSON.stringify(result)).not.toContain(rawResult);
      expect(JSON.stringify(message.toJSON())).not.toContain(rawResult);
    }
  );

  it.each(metadataFields)(
    'rejects nested %s accessors without invoking them',
    (field) => {
      const { canonical, input } = makeMessages(Message, 'designated-result');
      let reads = 0;
      const nested = Object.defineProperty({}, 'selectedResult', {
        enumerable: true,
        get: () => {
          reads++;
          return rawResult;
        },
      });
      canonical[field].nested = nested;

      expect(() =>
        makeGraph().validateToolResultSubagentResult(input, [canonical])
      ).toThrow(ToolResultProtectionError);
      expect(reads).toBe(0);
    }
  );

  it('detaches retained history and supplied message carriers and nested content', () => {
    const { canonical, supplied, input } = makeMessages(
      Message,
      'designated-result'
    );
    const result = makeGraph().validateToolResultSubagentResult(input, [
      canonical,
    ]);
    const before = JSON.stringify(result);
    const message = result.message as AIMessage | AIMessageChunk;
    expect(message.content).not.toBe(canonical.content);
    expect(message.tool_calls).not.toBe(canonical.tool_calls);
    expect(message.tool_calls?.[0]).not.toBe(canonical.tool_calls?.[0]);
    for (const field of metadataFields) {
      expect(message[field]).not.toBe(canonical[field]);
      expect(message[field]).not.toBe(supplied[field]);
      canonical[field].selectedResult = rawResult;
      supplied[field].selectedResult = rawResult;
    }
    for (const retained of [canonical, supplied]) {
      if (Array.isArray(retained.content)) retained.content[0].text = rawResult;
      retained.tool_calls![0].id = rawResult;
      retained.id = rawResult;
    }
    input.agentId = rawResult;
    input.message = new Message(rawResult);

    expect(JSON.stringify(result)).toBe(before);
    expect(JSON.stringify(message.toJSON())).not.toContain(rawResult);
    expect(message.lc_kwargs.content).toEqual(message.content);
  });

  it.each(['id', 'content', 'tool_calls'] as const)(
    'still rejects mismatched %s',
    (field) => {
      const { canonical, supplied, input } = makeMessages(
        Message,
        'designated-result'
      );
      if (field === 'id') supplied.id = 'other-result';
      if (field === 'content') supplied.content = rawResult;
      if (field === 'tool_calls') supplied.tool_calls = [];

      expect(() =>
        makeGraph().validateToolResultSubagentResult(input, [canonical])
      ).toThrow(ToolResultProtectionError);
    }
  );
});

it('preserves empty results and enforces the designated agent', () => {
  const graph = makeGraph();
  const empty = { agentId: 'result' };
  expect(graph.validateToolResultSubagentResult(empty, [], 'result')).toEqual(
    empty
  );
  expect(graph.validateToolResultSubagentResult(empty, [], 'result')).not.toBe(
    empty
  );
  expect(() =>
    graph.validateToolResultSubagentResult(empty, [], 'other')
  ).toThrow(ToolResultProtectionError);
});

it.each(['checkpoint', 'send'] as const)(
  'canonicalizes restored %s results',
  async (carrier) => {
    const { canonical, supplied, input } = makeMessages(
      AIMessage,
      'designated-result'
    );
    for (const field of metadataFields)
      supplied[field] = { selectedResult: rawResult };
    const state = { messages: [canonical], subagentResult: input };
    const graph = makeGraph();
    let result: SubagentGraphResult;
    if (carrier === 'checkpoint') {
      const restored = await graph.validateToolResultCheckpoint({
        values: state,
      });
      result = restored.values.subagentResult;
    } else {
      const restored = await graph.validateToolResultSend(
        new Send('result', state)
      );
      result = (restored.args as typeof state).subagentResult;
    }

    expectCanonical(result, canonical);
    expect(result.message).not.toBe(canonical);
  }
);

it.each(metadataFields)(
  'canonicalizes %s before Run forwards fresh inputs to the workflow',
  async (field) => {
    const gate: ToolResultProtection = {
      version: 1,
      toolNames: ['lookup'],
      timeoutMs: 1000,
      maxAttemptBytes: 65536,
      maxBufferedBytes: 262144,
      classify: () => 'prose',
      inspect: ({ content }) => ({
        version: 1,
        ok: true,
        value: {
          content: content.replaceAll(rawResult, '[EMAIL_1]'),
          replacements: 1,
          categories: [],
        },
      }),
    };
    const run = await Run.create({
      runId: `result-metadata-${field}`,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
        compileOptions: { checkpointer: new MemorySaver() },
      },
      toolResultProtection: gate,
    });
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['Allowed final answer'],
    });
    const canonical = new AIMessage({
      id: 'run-result',
      content: 'Allowed designated answer',
      additional_kwargs: { allowed: 'provider-detail' },
      response_metadata: { finish_reason: 'stop' },
    });
    const supplied = new AIMessage({
      id: canonical.id,
      content: canonical.content,
      additional_kwargs: { ...canonical.additional_kwargs },
      response_metadata: { ...canonical.response_metadata },
    });
    supplied[field] = { selectedResult: rawResult };
    const state = {
      messages: [new HumanMessage('Allowed input'), canonical],
      subagentResult: { agentId: 'result', message: supplied },
    };
    const stream = jest.spyOn(run.graphRunnable!, 'streamEvents');

    await run.processStream(state, {
      version: 'v2',
      configurable: { thread_id: `result-metadata-${field}` },
    });

    expect(stream).toHaveBeenCalledTimes(1);
    const forwarded = stream.mock.calls[0][0] as typeof state;
    expectCanonical(forwarded.subagentResult, canonical);
    expect(forwarded.subagentResult.message).not.toBe(canonical);
    expect(forwarded.subagentResult.message).not.toBe(supplied);
    expect(JSON.stringify(forwarded)).not.toContain(rawResult);
  }
);

import { describe, expect, it } from '@jest/globals';
import { HumanMessage } from '@langchain/core/messages';
import type {
  OpenAIPhysicalAttemptEvent,
  OpenAIPhysicalAttemptObserver,
} from './attemptObservation';
import { createOpenAIPhysicalAttemptFetch } from './attemptObservation';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatOpenAI } from '@/llm/openai';

type RecordedEvent = {
  attempt: unknown;
  event: OpenAIPhysicalAttemptEvent;
};

type StreamHooks = {
  onPull?: () => void;
  onCancel?: (reason: unknown) => void;
};

const encoder = new TextEncoder();
const privatePrompt = 'PHYSICAL_ATTEMPT_PRIVATE_PROMPT';
const privateCompletion = 'PHYSICAL_ATTEMPT_PRIVATE_COMPLETION';

function makeObserver() {
  const events: RecordedEvent[] = [];
  const attempt = { id: 'attempt-1' };
  const observer: OpenAIPhysicalAttemptObserver = {
    start: () => attempt,
    observe: (observedAttempt, event) => {
      events.push({ attempt: observedAttempt, event });
    },
  };
  return { attempt, events, observer };
}

function responseWithChunks(
  chunks: Uint8Array[],
  headers: HeadersInit = { 'content-type': 'text/event-stream' },
  hooks: StreamHooks = {},
): Response {
  let index = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      hooks.onPull?.();
      const chunk = chunks[index];
      if (chunk === undefined) {
        controller.close();
        return;
      }
      index += 1;
      controller.enqueue(chunk);
    },
    cancel(reason) {
      hooks.onCancel?.(reason);
    },
  }, { highWaterMark: 0 });
  return new Response(body, { status: 200, headers });
}

async function readResponseBytes(response: Response): Promise<Uint8Array> {
  const reader = response.body?.getReader();
  if (reader == null) {
    return new Uint8Array();
  }
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  while (true) {
    const result = await reader.read();
    if (result.done) {
      break;
    }
    if (result.value != null) {
      chunks.push(result.value);
      byteLength += result.value.byteLength;
    }
  }
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function successfulChatResponse(): Response {
  const frames = [
    {
      id: 'generation-native',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'openai/gpt-4.1',
      provider_name: 'upstream',
      choices: [{ index: 0, delta: { content: privateCompletion }, finish_reason: null }],
    },
    {
      id: 'generation-native',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'openai/gpt-4.1',
      provider_name: 'upstream',
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
        total_tokens: 3,
        cost: 0.01,
        total_cost: 0.02,
      },
    },
  ];
  const chunks = frames.map((frame) => encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
  chunks.push(encoder.encode('data: [DONE]\n\n'));
  return responseWithChunks(chunks, {
    'content-type': 'text/event-stream; charset=utf-8',
  });
}

function lastTerminal(events: RecordedEvent[]): Extract<
  OpenAIPhysicalAttemptEvent,
  { state: 'terminal' }
> | undefined {
  const terminals = events
    .map(({ event }) => event)
    .filter((event): event is Extract<OpenAIPhysicalAttemptEvent, { state: 'terminal' }> =>
      event.state === 'terminal',
    );
  return terminals[terminals.length - 1];
}

describe('OpenAI physical-attempt transport observation', () => {
  it('returns the original fetch when observation is disabled', () => {
    const baseFetch: typeof fetch = async () => new Response('ok');
    expect(createOpenAIPhysicalAttemptFetch({ baseFetch, provider: 'openai' })).toBe(baseFetch);
  });

  it('observes one unchanged fetch, bounded metadata, final stream charge, and exact response bytes', async () => {
    const usageEvent = JSON.stringify({
      id: 'generation-1',
      model: 'served/model',
      provider_name: 'upstream',
      choices: [{ delta: { content: `${privateCompletion} 🌍` } }],
      usage: {
        prompt_tokens: 7,
        completion_tokens: 3,
        total_tokens: 10,
        prompt_tokens_details: { cached_tokens: 2, cache_write_tokens: 1 },
        completion_tokens_details: { reasoning_tokens: 4 },
        cost: 0.01,
        total_cost: 0.02,
      },
    });
    const usageSplit = usageEvent.indexOf('"usage"');
    const wire =
      `event: message\r\ndata: ${usageEvent.slice(0, usageSplit)}\r\n` +
      `data: ${usageEvent.slice(usageSplit)}\r\rdata: [DONE]\n\n`;
    const sourceBytes = encoder.encode(wire);
    const unicodeOffset = sourceBytes.indexOf(0xf0);
    expect(unicodeOffset).toBeGreaterThan(0);
    const sourceChunks = [
      sourceBytes.slice(0, unicodeOffset + 2),
      sourceBytes.slice(unicodeOffset + 2),
    ];
    const { attempt, events, observer } = makeObserver();
    const input = 'https://provider.invalid/v1/chat/completions';
    const init: RequestInit = {
      method: 'POST',
      headers: { authorization: 'Bearer private-test-token' },
      body: privatePrompt,
    };
    let fetchCalls = 0;
    let forwardedInput: RequestInfo | URL | undefined;
    let forwardedInit: RequestInit | undefined;
    let sourcePulls = 0;
    const baseFetch: typeof fetch = async (requestInput, requestInit) => {
      fetchCalls += 1;
      forwardedInput = requestInput;
      forwardedInit = requestInit;
      return responseWithChunks(
        sourceChunks,
        {
          'content-type': 'text/event-stream',
          'x-provider-name': 'router-header',
        },
        { onPull: () => { sourcePulls += 1; } },
      );
    };
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch,
      provider: 'openrouter',
      observer,
    });

    const response = await observedFetch(input, init);
    expect(fetchCalls).toBe(1);
    expect(forwardedInput).toBe(input);
    expect(forwardedInit).toBe(init);
    expect(sourcePulls).toBe(0);
    expect(response.status).toBe(200);
    expect(response.headers.get('x-provider-name')).toBe('router-header');
    expect(new TextDecoder().decode(await readResponseBytes(response))).toBe(wire);
    expect(sourcePulls).toBeGreaterThan(0);
    expect(events.every((entry) => entry.attempt === attempt)).toBe(true);

    const eventValues = events.map(({ event }) => event);
    expect(eventValues.some((event) => event.state === 'accepted')).toBe(true);
    expect(eventValues.some((event) => event.state === 'first_output')).toBe(true);
    const charge = eventValues.find((event) => event.state === 'charge');
    expect(charge).toMatchObject({
      state: 'charge',
      routerCharge: { amountUsd: 0.02, source: 'router_final_usage' },
    });
    expect(eventValues.some((event) => event.state === 'charge_reconciled')).toBe(false);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'succeeded',
      servedProvider: 'upstream',
      servedModel: 'served/model',
      reconciliationGenerationId: 'generation-1',
      nativeUsage: {
        inputTokens: 7,
        outputTokens: 3,
        totalTokens: 10,
        cacheReadTokens: 2,
        cacheWriteTokens: 1,
        reasoningTokens: 4,
      },
      routerCharge: { amountUsd: 0.02, source: 'router_final_usage' },
    });
    const serializedEvents = JSON.stringify(events);
    expect(serializedEvents).not.toContain(privatePrompt);
    expect(serializedEvents).not.toContain(privateCompletion);
    expect(serializedEvents).not.toContain('private-test-token');
  });

  it('passes OpenRouter ClientOptions.fetch to the observed physical transport', async () => {
    const { events, observer } = makeObserver();
    let fetchCalls = 0;
    const baseFetch: typeof fetch = async () => {
      fetchCalls += 1;
      return successfulChatResponse();
    };
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch,
      provider: 'openrouter',
      observer,
    });
    const model = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      streaming: true,
      streamUsage: true,
      maxRetries: 0,
      _lc_stream_delay: 0,
      configuration: {
        baseURL: 'https://openrouter.invalid/api/v1',
        fetch: observedFetch,
      },
    });
    const content: string[] = [];
    const stream = await model.stream([new HumanMessage(privatePrompt)]);
    for await (const chunk of stream) {
      if (typeof chunk.content === 'string') {
        content.push(chunk.content);
      }
    }

    expect(content.join('')).toBe(privateCompletion);
    expect(fetchCalls).toBe(1);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'succeeded',
      nativeUsage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
      routerCharge: { amountUsd: 0.02, source: 'router_final_usage' },
    });
    expect(JSON.stringify(events)).not.toContain(privatePrompt);
    expect(JSON.stringify(events)).not.toContain(privateCompletion);
  });

  it('uses the same fetch seam for ChatOpenAI without reporting router charges', async () => {
    const { events, observer } = makeObserver();
    let fetchCalls = 0;
    const baseFetch: typeof fetch = async () => {
      fetchCalls += 1;
      return successfulChatResponse();
    };
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch,
      provider: 'openai',
      observer,
    });
    const model = new ChatOpenAI({
      model: 'gpt-4.1',
      apiKey: 'test-key',
      streaming: true,
      streamUsage: true,
      maxRetries: 0,
      configuration: {
        baseURL: 'https://api.openai.invalid/v1',
        fetch: observedFetch,
      },
    });
    const content: string[] = [];
    const stream = await model.stream([new HumanMessage(privatePrompt)]);
    for await (const chunk of stream) {
      if (typeof chunk.content === 'string') {
        content.push(chunk.content);
      }
    }

    expect(content.join('')).toBe(privateCompletion);
    expect(fetchCalls).toBe(1);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'succeeded',
      nativeUsage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
    });
    expect(events.some(({ event }) => event.state === 'charge')).toBe(false);
  });

  it('keeps unknown and provider-failed streams distinct when a DONE marker follows', async () => {
    const cases = [
      {
        payload: 'data: not-json\n\ndata: [DONE]\n\n',
        outcome: 'unknown',
        errorCode: 'unknown',
      },
      {
        payload: 'event: error\ndata: {"message":"private provider error"}\n\ndata: [DONE]\n\n',
        outcome: 'failed',
        errorCode: 'provider_error',
      },
      {
        payload: 'event: response.incomplete\ndata: {}\n\ndata: [DONE]\n\n',
        outcome: 'failed',
        errorCode: 'provider_error',
      },

      {
        payload: `data: ${JSON.stringify({ choices: [] })}\n\n`,
        outcome: 'unknown',
        errorCode: 'unknown',
      },
      {
        payload: `data: {"padding":"${'x'.repeat(16_384)}"}\n\ndata: [DONE]\n\n`,
        outcome: 'unknown',
        errorCode: 'unknown',
      },
    ] as const;

    for (const scenario of cases) {
      const { events, observer } = makeObserver();
      const wire = encoder.encode(scenario.payload);
      const observedFetch = createOpenAIPhysicalAttemptFetch({
        baseFetch: async () => responseWithChunks([wire]),
        provider: 'openai',
        observer,
      });
      const response = await observedFetch('https://provider.invalid');
      expect(new TextDecoder().decode(await readResponseBytes(response))).toBe(scenario.payload);
      expect(lastTerminal(events)).toMatchObject({
        state: 'terminal',
        outcome: scenario.outcome,
        errorCode: scenario.errorCode,
      });
      expect(JSON.stringify(events)).not.toContain('private provider error');
    }
  });

  it('recognizes Responses completion only after EOF and preserves its wire bytes', async () => {
    const payload = [
      'event: response.completed\n',
      `data: ${JSON.stringify({
        type: 'response.completed',
        response: {
          id: 'response-1',
          model: 'served/model',
          usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
        },
      })}\n\n`,
    ].join('');
    const { events, observer } = makeObserver();
    const bytes = encoder.encode(payload);
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch: async () => responseWithChunks([bytes]),
      provider: 'openai',
      observer,
    });

    const response = await observedFetch('https://provider.invalid');
    expect(new TextDecoder().decode(await readResponseBytes(response))).toBe(payload);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'succeeded',
      nativeUsage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
    });
  });

  it('classifies HTTP failures and rethrows transport errors unchanged', async () => {
    for (const [status, errorCode] of [
      [408, 'provider_timeout'],
      [504, 'provider_timeout'],
      [429, 'provider_rejected'],
      [503, 'provider_error'],
    ] as const) {
      const { events, observer } = makeObserver();
      const originalResponse = new Response('provider response body', { status });
      const observedFetch = createOpenAIPhysicalAttemptFetch({
        baseFetch: async () => originalResponse,
        provider: 'openrouter',
        observer,
      });
      const response = await observedFetch('https://provider.invalid');
      expect(response).toBe(originalResponse);
      expect(await response.text()).toBe('provider response body');
      expect(lastTerminal(events)).toMatchObject({
        state: 'terminal',
        outcome: 'failed',
        errorCode,
      });
    }

    const { events, observer } = makeObserver();
    const transportError = new Error('PHYSICAL_ATTEMPT_PRIVATE_TRANSPORT_ERROR');
    let fetchCalls = 0;
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch: async () => {
        fetchCalls += 1;
        throw transportError;
      },
      provider: 'openai',
      observer,
    });
    await expect(observedFetch('https://provider.invalid')).rejects.toBe(transportError);
    expect(fetchCalls).toBe(1);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'unknown',
      errorCode: 'unknown',
    });
    expect(JSON.stringify(events)).not.toContain(transportError.message);
  });

  it('preserves demand-driven byte identity, cancellation, and provider behavior when callbacks fail', async () => {
    const sourceChunk = encoder.encode('unchanged source bytes');
    let sourcePulls = 0;
    let sourceCancellation: unknown;
    const sourceBody = new ReadableStream<Uint8Array>({
      pull(controller) {
        sourcePulls += 1;
        controller.enqueue(sourceChunk);
      },
      cancel(reason) {
        sourceCancellation = reason;
      },
    }, { highWaterMark: 0 });
    const sourceResponse = new Response(sourceBody, {
      headers: { 'content-type': 'text/event-stream' },
    });
    const attempt = { id: 'attempt-cancel' };
    const events: RecordedEvent[] = [];
    const observer: OpenAIPhysicalAttemptObserver = {
      start: () => attempt,
      observe(observedAttempt, event) {
        events.push({ attempt: observedAttempt, event });
        if (event.state === 'accepted') {
          throw new Error('observer callback failure');
        }
        return Promise.reject(new Error('observer async failure'));
      },
    };
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch: async () => sourceResponse,
      provider: 'openai',
      observer,
    });

    const response = await observedFetch('https://provider.invalid');
    expect(sourcePulls).toBe(0);
    const reader = response.body?.getReader();
    expect(reader).toBeDefined();
    const first = await reader?.read();
    expect(first?.done).toBe(false);
    expect(first?.value).toBe(sourceChunk);
    expect(sourcePulls).toBe(1);
    const reason = { reason: 'consumer stopped' };
    await reader?.cancel(reason);
    expect(sourceCancellation).toBe(reason);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'unknown',
      errorCode: 'unknown',
    });
  });

  it('keeps the original stream read error and marks its outcome unknown', async () => {
    const streamError = new Error('PHYSICAL_ATTEMPT_PRIVATE_READ_ERROR');
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.error(streamError);
      },
    }, { highWaterMark: 0 });
    const sourceResponse = new Response(body, {
      headers: { 'content-type': 'text/event-stream' },
    });
    const { events, observer } = makeObserver();
    const observedFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch: async () => sourceResponse,
      provider: 'openai',
      observer,
    });

    const response = await observedFetch('https://provider.invalid');
    const reader = response.body?.getReader();
    if (reader == null) {
      throw new Error('Expected an observed response body');
    }
    await expect(reader.read()).rejects.toBe(streamError);
    expect(lastTerminal(events)).toMatchObject({
      state: 'terminal',
      outcome: 'unknown',
      errorCode: 'unknown',
    });
    expect(JSON.stringify(events)).not.toContain(streamError.message);
  });

  it('unwraps an earlier observer so nested wrappers count one physical dispatch', async () => {
    const first = makeObserver();
    const second = makeObserver();
    const payload = 'data: [DONE]\n\n';
    const bytes = encoder.encode(payload);
    let fetchCalls = 0;
    const baseFetch: typeof fetch = async () => {
      fetchCalls += 1;
      return responseWithChunks([bytes]);
    };
    const firstFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch,
      provider: 'openrouter',
      observer: first.observer,
    });
    const secondFetch = createOpenAIPhysicalAttemptFetch({
      baseFetch: firstFetch,
      provider: 'openrouter',
      observer: second.observer,
    });

    const response = await secondFetch('https://provider.invalid');
    expect(new TextDecoder().decode(await readResponseBytes(response))).toBe(payload);
    expect(fetchCalls).toBe(1);
    expect(first.events).toHaveLength(0);
    expect(lastTerminal(second.events)).toMatchObject({
      state: 'terminal',
      outcome: 'succeeded',
    });
  });
});

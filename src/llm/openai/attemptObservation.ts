export type OpenAIPhysicalAttemptProvider = 'openai' | 'openrouter';

export type OpenAIPhysicalAttemptErrorCode =
  | 'provider_error'
  | 'provider_timeout'
  | 'provider_rejected'
  | 'cancelled'
  | 'unknown';

export interface OpenAIPhysicalAttemptUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  reasoningTokens?: number;
}

export interface OpenAIPhysicalAttemptCharge {
  amountUsd: number;
  source: 'router_final_usage' | 'generation_reconciliation';
}

type AttemptRouteMetadata = {
  nativeUsage?: OpenAIPhysicalAttemptUsage;
  servedProvider?: string;
  servedModel?: string;
  reconciliationGenerationId?: string;
};

export type OpenAIPhysicalAttemptEvent =
  | { state: 'accepted' }
  | { state: 'first_output' }
  | (AttemptRouteMetadata & { state: 'route_metadata' })
  | (AttemptRouteMetadata & {
      state: 'charge';
      routerCharge: OpenAIPhysicalAttemptCharge & { source: 'router_final_usage' };
    })
  | (AttemptRouteMetadata & {
      state: 'charge_reconciled';
      routerCharge: OpenAIPhysicalAttemptCharge & { source: 'generation_reconciliation' };
      reconciliationGenerationId: string;
    })
  | (AttemptRouteMetadata & {
      state: 'terminal';
      outcome: 'succeeded' | 'failed' | 'cancelled' | 'unknown';
      errorCode?: OpenAIPhysicalAttemptErrorCode;
      routerCharge?: OpenAIPhysicalAttemptCharge;
    });

/** Content-free callbacks for one physical OpenAI-compatible HTTP request. */
export interface OpenAIPhysicalAttemptObserver {
  start(): unknown;
  observe(
    attempt: unknown,
    event: OpenAIPhysicalAttemptEvent,
  ): void | Promise<void>;
}

const MAX_SSE_LINE_LENGTH = 16_384;
const MAX_SSE_EVENT_LENGTH = 16_384;
const SAFE_PROVIDER_LABEL = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const SAFE_MODEL_LABEL = /^[A-Za-z0-9][A-Za-z0-9._/+@-]{0,159}$/;
const SAFE_GENERATION_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const FETCH_BASES = new WeakMap<typeof fetch, typeof fetch>();

type EventRecord = Record<string, unknown>;

function isRecord(value: unknown): value is EventRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finiteNonnegative(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function safeLabel(value: unknown, pattern: RegExp): string | undefined {
  return typeof value === 'string' && pattern.test(value) ? value : undefined;
}

function safeGenerationId(value: unknown): string | undefined {
  return safeLabel(value, SAFE_GENERATION_ID);
}

function firstNumber(
  first: unknown,
  second?: unknown,
  third?: unknown,
): number | undefined {
  const firstValue = finiteNonnegative(first);
  if (firstValue !== undefined) {
    return firstValue;
  }
  const secondValue = finiteNonnegative(second);
  if (secondValue !== undefined) {
    return secondValue;
  }
  return finiteNonnegative(third);
}

function projectUsage(value: unknown): OpenAIPhysicalAttemptUsage | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const promptDetails = isRecord(value.prompt_tokens_details)
    ? value.prompt_tokens_details
    : undefined;
  const inputDetails = isRecord(value.input_tokens_details) ? value.input_tokens_details : undefined;
  const completionDetails = isRecord(value.completion_tokens_details)
    ? value.completion_tokens_details
    : undefined;
  const outputDetails = isRecord(value.output_tokens_details) ? value.output_tokens_details : undefined;
  const inputTokens = firstNumber(value.prompt_tokens, value.input_tokens);
  const outputTokens = firstNumber(value.completion_tokens, value.output_tokens);
  const totalTokens = finiteNonnegative(value.total_tokens);
  const cacheReadTokens = firstNumber(
    promptDetails?.cached_tokens,
    inputDetails?.cached_tokens,
    value.cache_read_tokens,
  );
  const cacheWriteTokens = firstNumber(
    promptDetails?.cache_write_tokens,
    promptDetails?.cache_creation_tokens,
    inputDetails?.cache_write_tokens ??
      inputDetails?.cache_creation_tokens ??
      value.cache_write_tokens,
  );
  const reasoningTokens = firstNumber(
    completionDetails?.reasoning_tokens,
    outputDetails?.reasoning_tokens,
    value.reasoning_tokens,
  );
  const usage: OpenAIPhysicalAttemptUsage = {};
  if (inputTokens !== undefined) {
    usage.inputTokens = inputTokens;
  }
  if (outputTokens !== undefined) {
    usage.outputTokens = outputTokens;
  }
  if (totalTokens !== undefined) {
    usage.totalTokens = totalTokens;
  }
  if (cacheReadTokens !== undefined) {
    usage.cacheReadTokens = cacheReadTokens;
  }
  if (cacheWriteTokens !== undefined) {
    usage.cacheWriteTokens = cacheWriteTokens;
  }
  if (reasoningTokens !== undefined) {
    usage.reasoningTokens = reasoningTokens;
  }
  return inputTokens === undefined &&
    outputTokens === undefined &&
    totalTokens === undefined &&
    cacheReadTokens === undefined &&
    cacheWriteTokens === undefined &&
    reasoningTokens === undefined
    ? undefined
    : usage;
}

function mergeUsage(
  current: OpenAIPhysicalAttemptUsage | undefined,
  next: OpenAIPhysicalAttemptUsage | undefined,
): OpenAIPhysicalAttemptUsage | undefined {
  if (next === undefined) {
    return current;
  }
  return { ...current, ...next };
}

function statusErrorCode(status: number): OpenAIPhysicalAttemptErrorCode {
  if (status === 408 || status === 504) {
    return 'provider_timeout';
  }
  if (status >= 400 && status < 500) {
    return 'provider_rejected';
  }
  if (status >= 500) {
    return 'provider_error';
  }
  return 'unknown';
}

function createBoundedSseEventParser(
  onEvent: (eventName: string, data: string) => void,
  onOverflow: () => void,
): { push(text: string): void } {
  let lineParts: string[] = [];
  let lineLength = 0;
  let lineOverflow = false;
  let skipLineFeed = false;
  let eventLength = 0;
  let eventOverflow = false;
  let eventName = '';
  let dataLines: string[] = [];
  let hasData = false;

  const resetEvent = (): void => {
    eventLength = 0;
    eventOverflow = false;
    eventName = '';
    dataLines = [];
    hasData = false;
  };
  const dispatchEvent = (): void => {
    if (eventOverflow) {
      onOverflow();
    } else if (hasData || eventName !== '') {
      onEvent(eventName, dataLines.join('\n'));
    }
    resetEvent();
  };
  const finishLine = (): void => {
    const blankLine = lineLength === 0 && !lineOverflow;
    if (lineOverflow || eventLength + lineLength + 1 > MAX_SSE_EVENT_LENGTH) {
      eventOverflow = true;
    } else {
      eventLength += lineLength + 1;
    }
    if (blankLine) {
      dispatchEvent();
    } else if (!lineOverflow && !eventOverflow) {
      const line = lineParts.join('');
      if (!line.startsWith(':')) {
        const separator = line.indexOf(':');
        const field = separator === -1 ? line : line.slice(0, separator);
        let value = separator === -1 ? '' : line.slice(separator + 1);
        if (value.startsWith(' ')) {
          value = value.slice(1);
        }
        if (field === 'event') {
          eventName = value;
        } else if (field === 'data') {
          hasData = true;
          dataLines.push(value);
        }
      }
    }
    lineParts = [];
    lineLength = 0;
    lineOverflow = false;
  };
  const appendLineText = (text: string): void => {
    if (lineOverflow || text.length === 0) {
      return;
    }
    if (lineLength + text.length > MAX_SSE_LINE_LENGTH) {
      lineOverflow = true;
      lineParts = [];
      return;
    }
    lineParts.push(text);
    lineLength += text.length;
  };

  return {
    push(text: string): void {
      if (text.length === 0) {
        return;
      }
      let offset = 0;
      if (skipLineFeed) {
        skipLineFeed = false;
        if (text.charAt(0) === '\n') {
          offset = 1;
        }
      }
      while (offset < text.length) {
        const lineFeed = text.indexOf('\n', offset);
        const carriageReturn = text.indexOf('\r', offset);
        const end = Math.min(
          lineFeed === -1 ? text.length : lineFeed,
          carriageReturn === -1 ? text.length : carriageReturn,
        );
        appendLineText(text.slice(offset, end));
        if (end === text.length) {
          return;
        }
        const isCarriageReturn = text.charAt(end) === '\r';
        finishLine();
        offset = end + 1;
        if (isCarriageReturn) {
          if (text.charAt(offset) === '\n') {
            offset += 1;
          } else if (offset === text.length) {
            skipLineFeed = true;
          }
        }
      }
    },
  };
}

function hasNonemptyText(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.length > 0;
  }
  return (
    Array.isArray(value) &&
    value.some(
      (part) =>
        isRecord(part) &&
        typeof part.text === 'string' &&
        part.text.length > 0,
    )
  );
}

function hasSemanticOutput(parsed: EventRecord, eventName: string): boolean {
  if (eventName.startsWith('response.') && eventName.endsWith('.delta')) {
    return hasNonemptyText(parsed.delta);
  }
  if (eventName === 'response.output_item.added' && isRecord(parsed.item)) {
    return (
      parsed.item.type === 'function_call' &&
      (hasNonemptyText(parsed.item.name) || hasNonemptyText(parsed.item.call_id))
    );
  }
  if (!Array.isArray(parsed.choices)) {
    return false;
  }
  return parsed.choices.some((choice) => {
    if (!isRecord(choice) || !isRecord(choice.delta)) {
      return false;
    }
    const delta = choice.delta;
    if (
      hasNonemptyText(delta.content) ||
      hasNonemptyText(delta.reasoning) ||
      hasNonemptyText(delta.reasoning_content)
    ) {
      return true;
    }
    return (
      Array.isArray(delta.tool_calls) &&
      delta.tool_calls.some((toolCall) => {
        if (!isRecord(toolCall)) {
          return false;
        }
        const fn = isRecord(toolCall.function) ? toolCall.function : undefined;
        return (
          hasNonemptyText(toolCall.id) ||
          hasNonemptyText(fn?.name) ||
          hasNonemptyText(fn?.arguments)
        );
      })
    );
  });
}

function notifyObserver(
  observer: OpenAIPhysicalAttemptObserver,
  attempt: unknown,
  event: OpenAIPhysicalAttemptEvent,
): void {
  try {
    const result = observer.observe(attempt, event);
    if (result != null) {
      void result.catch(() => undefined);
    }
  } catch {
    // Telemetry callbacks are not part of provider request behavior.
  }
}

function responseWithObservedBody(
  response: Response,
  body: ReadableStream<Uint8Array>,
): Response {
  const observed = new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
  Object.defineProperties(observed, {
    url: { value: response.url, enumerable: true },
    redirected: { value: response.redirected, enumerable: true },
    type: { value: response.type, enumerable: true },
  });
  return observed;
}

/**
 * Wraps the OpenAI SDK's supported `fetch` option. Each invocation records one
 * physical dispatch; this observer does not change or add retry behavior.
 */
export function createOpenAIPhysicalAttemptFetch({
  baseFetch,
  provider,
  observer,
}: {
  baseFetch: typeof fetch;
  provider: OpenAIPhysicalAttemptProvider;
  observer?: OpenAIPhysicalAttemptObserver;
}): typeof fetch {
  if (observer == null) {
    return baseFetch;
  }

  const fetchBase = FETCH_BASES.get(baseFetch) ?? baseFetch;

  const observedFetch: typeof fetch = async (input, init) => {
    let attempt: unknown;
    let observerStarted = false;
    try {
      attempt = observer.start();
      observerStarted = true;
    } catch {
      // Observer setup must not prevent the provider request.
    }

    let terminalSent = false;
    let nativeUsage: OpenAIPhysicalAttemptUsage | undefined;
    let servedProvider: string | undefined;
    let servedModel: string | undefined;
    let routerCharge: OpenAIPhysicalAttemptCharge | undefined;
    let reconciliationGenerationId: string | undefined;
    let streamChargeIncludesTotalCost = false;
    let streamCompleted = false;
    let doneMarkerSeen = false;
    let streamOutcome: 'failed' | 'unknown' | undefined;
    let firstOutputObserved = false;

    const notify = (event: OpenAIPhysicalAttemptEvent): void => {
      if (observerStarted) {
        notifyObserver(observer, attempt, event);
      }
    };
    const notifyTerminal = (
      outcome: 'succeeded' | 'failed' | 'unknown',
      errorCode?: OpenAIPhysicalAttemptErrorCode,
    ): void => {
      if (terminalSent) {
        return;
      }
      terminalSent = true;
      notify({
        state: 'terminal',
        outcome,
        ...(errorCode === undefined ? {} : { errorCode }),
        ...(nativeUsage === undefined ? {} : { nativeUsage }),
        ...(servedProvider === undefined ? {} : { servedProvider }),
        ...(servedModel === undefined ? {} : { servedModel }),
        ...(routerCharge === undefined ? {} : { routerCharge }),
        ...(reconciliationGenerationId === undefined
          ? {}
          : { reconciliationGenerationId }),
      });
    };
    const notifyRouteMetadata = (): void => {
      if (
        servedProvider === undefined &&
        servedModel === undefined &&
        reconciliationGenerationId === undefined
      ) {
        return;
      }
      notify({
        state: 'route_metadata',
        ...(servedProvider === undefined ? {} : { servedProvider }),
        ...(servedModel === undefined ? {} : { servedModel }),
        ...(reconciliationGenerationId === undefined
          ? {}
          : { reconciliationGenerationId }),
      });
    };
    const markUnknown = (): void => {
      if (streamOutcome !== 'failed') {
        streamOutcome = 'unknown';
      }
    };
    const markProviderFailure = (): void => {
      streamOutcome = 'failed';
      notifyTerminal('failed', 'provider_error');
    };

    let response: Response;
    try {
      response = await fetchBase.call(undefined, input, init);
    } catch (error) {
      streamOutcome = 'unknown';
      notifyTerminal('unknown', 'unknown');
      throw error;
    }

    if (!response.ok) {
      notifyTerminal('failed', statusErrorCode(response.status));
      return response;
    }

    notify({ state: 'accepted' });
    const responseBody = response.body;
    if (responseBody == null) {
      notifyTerminal('unknown', 'unknown');
      return response;
    }

    const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
    const isEventStream = contentType.includes('text/event-stream');
    servedProvider = safeLabel(
      response.headers.get('x-provider-name') ?? response.headers.get('provider-name'),
      SAFE_PROVIDER_LABEL,
    );
    notifyRouteMetadata();

    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    let decoder: TextDecoder | undefined;

    const updateMetadata = (parsed: EventRecord): void => {
      const responseData = isRecord(parsed.response) ? parsed.response : parsed;
      let rawUsage: EventRecord | undefined;
      if (isRecord(responseData.usage)) {
        rawUsage = responseData.usage;
      } else if (isRecord(parsed.usage)) {
        rawUsage = parsed.usage;
      }
      const incomingGenerationId =
        provider === 'openrouter'
          ? safeGenerationId(responseData.id ?? parsed.id)
          : undefined;
      if (
        incomingGenerationId !== undefined &&
        reconciliationGenerationId !== undefined &&
        incomingGenerationId !== reconciliationGenerationId
      ) {
        return;
      }

      const previousProvider = servedProvider;
      const previousModel = servedModel;
      const previousGenerationId = reconciliationGenerationId;
      servedModel =
        safeLabel(responseData.model, SAFE_MODEL_LABEL) ??
        safeLabel(parsed.model, SAFE_MODEL_LABEL) ??
        servedModel;
      servedProvider =
        safeLabel(responseData.provider_name, SAFE_PROVIDER_LABEL) ??
        safeLabel(parsed.provider_name, SAFE_PROVIDER_LABEL) ??
        safeLabel(responseData.provider, SAFE_PROVIDER_LABEL) ??
        safeLabel(parsed.provider, SAFE_PROVIDER_LABEL) ??
        servedProvider;
      reconciliationGenerationId ??= incomingGenerationId;
      if (
        servedProvider !== previousProvider ||
        servedModel !== previousModel ||
        reconciliationGenerationId !== previousGenerationId
      ) {
        notifyRouteMetadata();
      }

      if (rawUsage !== undefined) {
        nativeUsage = mergeUsage(nativeUsage, projectUsage(rawUsage));
      }
      if (provider !== 'openrouter') {
        return;
      }

      const totalCost = firstNumber(
        rawUsage?.total_cost,
        responseData.total_cost,
        parsed.total_cost,
      );
      const finalCost = finiteNonnegative(rawUsage?.cost);
      if (totalCost !== undefined) {
        streamChargeIncludesTotalCost = true;
      }
      const amountUsd =
        totalCost ??
        (streamChargeIncludesTotalCost ? undefined : finalCost);
      if (amountUsd === undefined) {
        return;
      }

      const nextCharge: Extract<
        OpenAIPhysicalAttemptEvent,
        { state: 'charge' }
      >['routerCharge'] = {
        amountUsd,
        source: 'router_final_usage',
      };
      const unchanged =
        routerCharge?.source === nextCharge.source &&
        routerCharge.amountUsd === nextCharge.amountUsd;
      routerCharge = nextCharge;
      if (!unchanged) {
        notify({
          state: 'charge',
          routerCharge: nextCharge,
          ...(reconciliationGenerationId === undefined
            ? {}
            : { reconciliationGenerationId }),
          ...(nativeUsage === undefined ? {} : { nativeUsage }),
          ...(servedProvider === undefined ? {} : { servedProvider }),
          ...(servedModel === undefined ? {} : { servedModel }),
        });
      }
    };

    const observeSseEvent = (eventName: string, data: string): void => {
      if (doneMarkerSeen) {
        return;
      }
      if (
        eventName === 'error' ||
        eventName === 'response.failed' ||
        eventName === 'response.incomplete'
      ) {
        markProviderFailure();
        return;
      }
      if (data.startsWith('[DONE]')) {
        streamCompleted = true;
        doneMarkerSeen = true;
        return;
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        markUnknown();
        return;
      }
      if (!isRecord(parsed)) {
        markUnknown();
        return;
      }
      const parsedEventName =
        eventName || (typeof parsed.type === 'string' ? parsed.type : '');
      if (
        parsedEventName === 'error' ||
        parsedEventName === 'response.failed' ||
        parsedEventName === 'response.incomplete' ||
        parsed.error != null
      ) {
        markProviderFailure();
        return;
      }

      updateMetadata(parsed);
      if (!firstOutputObserved && hasSemanticOutput(parsed, parsedEventName)) {
        firstOutputObserved = true;
        notify({ state: 'first_output' });
      }
      if (parsedEventName === 'response.completed') {
        streamCompleted = true;
      }
    };
    const sseParser = createBoundedSseEventParser(observeSseEvent, markUnknown);
    const consumeSseText = (text: string): void => {
      try {
        sseParser.push(text);
      } catch {
        markUnknown();
      }
    };

    const observedBody = new ReadableStream<Uint8Array>({
      async pull(controller): Promise<void> {
        let result: ReadableStreamReadResult<Uint8Array>;
        try {
          reader ??= responseBody.getReader();
          result = await reader.read();
        } catch (error) {
          if (streamOutcome !== 'failed') {
            markUnknown();
          }
          notifyTerminal(
            streamOutcome === 'failed' ? 'failed' : 'unknown',
            streamOutcome === 'failed' ? 'provider_error' : 'unknown',
          );
          controller.error(error);
          return;
        }

        if (result.done) {
          if (isEventStream && decoder != null) {
            try {
              consumeSseText(decoder.decode());
            } catch {
              markUnknown();
            }
          }
          if (!terminalSent) {
            if (streamOutcome === 'failed') {
              notifyTerminal('failed', 'provider_error');
            } else if (
              isEventStream &&
              streamCompleted &&
              streamOutcome === undefined
            ) {
              notifyTerminal('succeeded');
            } else {
              markUnknown();
              notifyTerminal('unknown', 'unknown');
            }
          }
          controller.close();
          return;
        }

        if (result.value == null) {
          return;
        }
        if (isEventStream) {
          decoder ??= new TextDecoder();
          try {
            consumeSseText(decoder.decode(result.value, { stream: true }));
          } catch {
            markUnknown();
          }
        }
        controller.enqueue(result.value);
      },
      async cancel(reason): Promise<void> {
        if (streamOutcome !== 'failed') {
          markUnknown();
        }
        notifyTerminal(
          streamOutcome === 'failed' ? 'failed' : 'unknown',
          streamOutcome === 'failed' ? 'provider_error' : 'unknown',
        );
        if (reader != null) {
          return reader.cancel(reason);
        }
        return responseBody.cancel(reason);
      },
    }, { highWaterMark: 0 });

    try {
      return responseWithObservedBody(response, observedBody);
    } catch {
      notifyTerminal('unknown', 'unknown');
      return response;
    }
  };

  FETCH_BASES.set(observedFetch, fetchBase);
  return observedFetch;
}

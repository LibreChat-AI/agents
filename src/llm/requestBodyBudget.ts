import { stampRetryable } from '@langchain/core/errors';

/** Local serialization failures must not enter transport retries or fallbacks. */
export class RequestBodyBudgetError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RequestBodyBudgetError';
    stampRetryable(this, false);
  }
}

/** Byte measurements are local transport constraints, not provider token limits. */
export class RequestBodyLimitError extends RequestBodyBudgetError {
  readonly source = 'local_request_body';

  constructor(
    readonly observedBytes: number,
    readonly limitBytes: number
  ) {
    super(
      `Serialized request body is ${observedBytes} bytes; local limit is ${limitBytes} bytes.`
    );
    this.name = 'RequestBodyLimitError';
  }
}

export function validateRequestBodyLimit(limit?: number): number | undefined {
  if (limit != null && (!Number.isSafeInteger(limit) || limit <= 0)) {
    throw new RangeError(
      'maxRequestBodyBytes must be a positive safe integer.'
    );
  }
  return limit;
}

/** Inspect the SDK-built body without consuming streams or rewriting JSON. */
export function assertRequestBodyBudget(
  body: RequestInit['body'],
  limit?: number
): void {
  if (limit == null || body == null) {
    return;
  }
  if (typeof body !== 'string') {
    throw new RequestBodyBudgetError(
      'maxRequestBodyBytes requires a serialized string request body.'
    );
  }
  const bytes = Buffer.byteLength(body, 'utf8');
  if (bytes > limit) {
    throw new RequestBodyLimitError(bytes, limit);
  }
}

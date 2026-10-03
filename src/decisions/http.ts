import type {
  DecisionModel,
  DecisionAnswer,
  DecisionResult,
  DecisionRequest,
  DecisionDialect,
  DecisionCredential,
  DecisionQuestion,
} from './types';
import type { DecisionFetch } from './transport';
import { DecisionError, isDecisionObject, readDecisionUsage } from './types';
import { validateDecisionQuestions } from './questions';
import { toWireQuestion, readAnswer } from './dialect';
import { createTransport } from './transport';

export const HTTP_PROVIDER_ID = 'http';

export interface HttpDecisionModelOptions {
  providerId?: string;
  apiKey?: DecisionCredential;
  requiresAuth?: boolean;
  /** Full URL, not a base path. */
  endpoint: string;
  model?: string;
  dialect?: DecisionDialect;
  requestKey?: string;
  responseKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetch?: DecisionFetch;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  onAnswered?: (label: string, ms: number) => void;
}

export function parseEnvelope(
  body: string,
  providerId: string,
  readOne: (
    answer: unknown,
    question?: DecisionQuestion
  ) => DecisionAnswer | null,
  responseKey?: string,
  expected?: Record<string, DecisionQuestion>
): DecisionResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    throw new DecisionError('malformed_response', 'response was not JSON', {
      provider: providerId,
    });
  }
  if (!isDecisionObject(parsed)) {
    throw new DecisionError(
      'malformed_response',
      'response was not an object',
      {
        provider: providerId,
      }
    );
  }
  const unwrapped =
    responseKey != null && responseKey !== '' ? parsed[responseKey] : parsed;
  if (
    !isDecisionObject(unwrapped) ||
    !isDecisionObject(unwrapped.answers) ||
    Object.keys(unwrapped.answers).length === 0
  ) {
    throw new DecisionError(
      'malformed_response',
      'response carried no answers',
      {
        provider: providerId,
      }
    );
  }

  const answers: DecisionResult['answers'] = Object.create(null);
  for (const [id, answer] of Object.entries(unwrapped.answers)) {
    if (expected && !Object.hasOwn(expected, id)) {
      throw new DecisionError(
        'malformed_response',
        'response carried an unknown answer',
        {
          provider: providerId,
        }
      );
    }
    const mapped = readOne(answer, expected?.[id]);
    if (!mapped) {
      throw new DecisionError(
        'malformed_response',
        'response carried an invalid answer',
        {
          provider: providerId,
        }
      );
    }
    answers[id] = mapped;
  }

  return {
    model: typeof unwrapped.model === 'string' ? unwrapped.model : 'unknown',
    answers,
    usage: readDecisionUsage(unwrapped.usage),
  };
}

/** Jev and Laya use the same System One HTTP dialect, but may report different confidence metrics. */
export function createHttpDecisionModel(
  options: HttpDecisionModelOptions
): DecisionModel {
  const providerId = options.providerId ?? HTTP_PROVIDER_ID;
  const send = createTransport({ ...options, providerId });
  const model = options.model ?? '';
  const dialect: DecisionDialect = options.dialect ?? 'port';
  const { requestKey, responseKey } = options;

  return {
    id: providerId,
    model,
    async decide(request: DecisionRequest): Promise<DecisionResult> {
      return send(
        () => {
          const entries = validateDecisionQuestions(
            request.questions,
            providerId
          );
          const questions: Record<
            string,
            ReturnType<typeof toWireQuestion>
          > = Object.create(null);
          const expected: Record<string, DecisionQuestion> =
            Object.create(null);
          for (const [id, question] of entries) {
            questions[id] = toWireQuestion(question, dialect);
            expected[id] = question;
          }
          const inner = { state: request.state, questions };
          let payload: string;
          try {
            payload = JSON.stringify({
              ...(model !== '' ? { model } : {}),
              ...(requestKey != null && requestKey !== ''
                ? { [requestKey]: inner }
                : inner),
            });
          } catch {
            throw new DecisionError(
              'bad_request',
              'invalid decision model request',
              {
                provider: providerId,
              }
            );
          }
          return {
            payload,
            parse: (body: string) =>
              parseEnvelope(
                body,
                providerId,
                (answer, question) => readAnswer(answer, dialect, question),
                responseKey,
                expected
              ),
          };
        },
        request.signal,
        request.label ?? 'decide',
        request.timeoutMs
      );
    },
  };
}

import type {
  Classifier,
  ClassificationAnswer,
  ClassificationResult,
  ClassificationRequest,
  ClassificationDialect,
  ClassificationCredential,
  ClassificationQuestion,
} from './types';
import type { ClassificationFetch } from './transport';
import {
  ClassificationError,
  isClassificationObject,
  readClassificationUsage,
} from './types';
import { validateClassificationQuestions } from './questions';
import { toWireQuestion, readAnswer } from './dialect';
import { createTransport } from './transport';

export const HTTP_PROVIDER_ID = 'http';

export interface HttpClassifierOptions {
  providerId?: string;
  apiKey?: ClassificationCredential;
  requiresAuth?: boolean;
  /** Full URL, not a base path. */
  endpoint: string;
  model?: string;
  dialect?: ClassificationDialect;
  requestKey?: string;
  responseKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetch?: ClassificationFetch;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  onAnswered?: (label: string, ms: number) => void;
}

export function parseEnvelope(
  body: string,
  providerId: string,
  readOne: (
    answer: unknown,
    question?: ClassificationQuestion
  ) => ClassificationAnswer | null,
  responseKey?: string,
  expected?: Record<string, ClassificationQuestion>
): ClassificationResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    throw new ClassificationError(
      'malformed_response',
      'response was not JSON',
      {
        provider: providerId,
      }
    );
  }
  if (!isClassificationObject(parsed)) {
    throw new ClassificationError(
      'malformed_response',
      'response was not an object',
      {
        provider: providerId,
      }
    );
  }
  const unwrapped = responseKey != null && responseKey !== '' ? parsed[responseKey] : parsed;
  if (
    !isClassificationObject(unwrapped) ||
    !isClassificationObject(unwrapped.answers) ||
    Object.keys(unwrapped.answers).length === 0
  ) {
    throw new ClassificationError(
      'malformed_response',
      'response carried no answers',
      {
        provider: providerId,
      }
    );
  }

  const answers: ClassificationResult['answers'] = Object.create(null);
  for (const [id, answer] of Object.entries(unwrapped.answers)) {
    if (expected && !Object.hasOwn(expected, id)) {
      throw new ClassificationError(
        'malformed_response',
        'response carried an unknown answer',
        {
          provider: providerId,
        }
      );
    }
    const mapped = readOne(answer, expected?.[id]);
    if (!mapped) {
      throw new ClassificationError(
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
    usage: readClassificationUsage(unwrapped.usage),
  };
}

/** Jev and Laya use the same System One HTTP dialect, but may report different confidence metrics. */
export function createHttpClassifier(
  options: HttpClassifierOptions
): Classifier {
  const providerId = options.providerId ?? HTTP_PROVIDER_ID;
  const send = createTransport({ ...options, providerId });
  const model = options.model ?? '';
  const dialect: ClassificationDialect = options.dialect ?? 'port';
  const { requestKey, responseKey } = options;

  return {
    id: providerId,
    model,
    async classify(
      request: ClassificationRequest
    ): Promise<ClassificationResult> {
      return send(
        () => {
          const entries = validateClassificationQuestions(
            request.questions,
            providerId
          );
          const questions: Record<
            string,
            ReturnType<typeof toWireQuestion>
          > = Object.create(null);
          const expected: Record<string, ClassificationQuestion> =
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
              ...(requestKey != null && requestKey !== '' ? { [requestKey]: inner } : inner),
            });
          } catch {
            throw new ClassificationError(
              'bad_request',
              'invalid classifier request',
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
        request.label ?? 'classify',
        request.timeoutMs
      );
    },
  };
}

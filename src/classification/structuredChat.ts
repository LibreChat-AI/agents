import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type {
  Classifier,
  ClassificationQuestion,
  ClassificationResult,
  ClassificationRequest,
} from './types';
import {
  ClassificationError,
  isClassificationObject,
  readClassificationUsage,
} from './types';
import { validateClassificationQuestions } from './questions';
import { CLASSIFICATION_PROMPT_PREFIX } from './traceMarker';
import { failureForStatus, retryAfterMs } from './transport';
import { withClassificationDeadline } from './deadline';

const MAX_QUESTIONS = 32;
const MAX_CHOICE_OPTIONS = 128;
const DEFAULT_TIMEOUT_MS = 20_000;
const PROVIDER_ID = 'structured-chat';

function readChatUsage(raw: unknown): ClassificationResult['usage'] {
  if (!isClassificationObject(raw)) {
    return null;
  }
  const usage = readClassificationUsage(raw.usage_metadata);
  const response = isClassificationObject(raw.response_metadata)
    ? raw.response_metadata
    : null;
  const nested = isClassificationObject(response?.metadata)
    ? response.metadata
    : null;
  const bedrock = isClassificationObject(nested?.usage)
    ? nested.usage
    : response?.usage;
  if (
    usage?.inputTokens === undefined ||
    !isClassificationObject(bedrock) ||
    usage.inputTokens !== bedrock.inputTokens
  ) {
    return usage;
  }
  const read = bedrock.cacheReadInputTokens;
  const write = bedrock.cacheWriteInputTokens;
  if (
    (read != null &&
      (typeof read !== 'number' || !Number.isSafeInteger(read) || read < 0)) ||
    (write != null &&
      (typeof write !== 'number' || !Number.isSafeInteger(write) || write < 0))
  ) {
    return usage;
  }
  const inputTokens =
    usage.inputTokens +
    (typeof read === 'number' ? read : 0) +
    (typeof write === 'number' ? write : 0);
  return Number.isSafeInteger(inputTokens) ? { ...usage, inputTokens } : usage;
}

function providerFailure(
  error: unknown,
  provider: string
): ClassificationError {
  const details = isClassificationObject(error) ? error : null;
  const metadata = isClassificationObject(details?.$metadata)
    ? details.$metadata
    : null;
  const rawStatus =
    details?.status ?? details?.statusCode ?? metadata?.httpStatusCode;
  const status =
    typeof rawStatus === 'number' &&
    Number.isInteger(rawStatus) &&
    rawStatus >= 100 &&
    rawStatus <= 599
      ? rawStatus
      : undefined;
  const classified = new ClassificationError(
    status == null ? 'network' : failureForStatus(status),
    'structured classifier request failed',
    { provider, status }
  );
  const headers = details?.headers;
  if (headers instanceof Headers) {
    classified.retryAfterMs = retryAfterMs(headers.get('retry-after'));
  }
  return classified;
}

export interface StructuredChatClassifierOptions {
  /** An already configured chat model whose chosen method enforces strict schemas. */
  model: BaseChatModel;
  modelId: string;
  providerId?: string;
  /** Choose the mode verified for this provider; JSON-mode prompting is never a fallback. */
  method: 'jsonSchema' | 'functionCalling';
  timeoutMs?: number;
  maxQuestions?: number;
  onAnswered?: (label: string, ms: number) => void;
}

function decisionSchema(
  entries: Array<[string, ClassificationQuestion]>
): Record<string, unknown> {
  const properties: Record<string, object> = Object.create(null);
  for (const [id, question] of entries) {
    if (question.type === 'boolean') {
      properties[id] = {
        type: 'object',
        properties: { decision: { type: 'boolean' } },
        required: ['decision'],
        additionalProperties: false,
      };
      continue;
    }
    properties[id] = {
      type: 'object',
      properties: {
        choice: { type: 'string', enum: Object.keys(question.criteria) },
      },
      required: ['choice'],
      additionalProperties: false,
    };
  }
  return {
    type: 'object',
    properties: {
      answers: {
        type: 'object',
        properties,
        required: entries.map(([id]) => id),
        additionalProperties: false,
      },
    },
    required: ['answers'],
    additionalProperties: false,
  };
}

/** Do not trust even a provider-parsed response: check every required id and choice locally. */
function readDecisions(
  parsed: unknown,
  entries: Array<[string, ClassificationQuestion]>,
  provider: string
): ClassificationResult['answers'] {
  const invalid = (): ClassificationError =>
    new ClassificationError(
      'malformed_response',
      'invalid structured classifier answer',
      {
        provider,
      }
    );
  if (
    !isClassificationObject(parsed) ||
    Object.keys(parsed).length !== 1 ||
    !isClassificationObject(parsed.answers) ||
    Object.keys(parsed.answers).length !== entries.length
  ) {
    throw invalid();
  }
  const answers: ClassificationResult['answers'] = Object.create(null);
  for (const [id, question] of entries) {
    if (!Object.hasOwn(parsed.answers, id)) {
      throw invalid();
    }
    const raw = parsed.answers[id];
    if (!isClassificationObject(raw) || Object.keys(raw).length !== 1) {
      throw invalid();
    }
    if (question.type === 'boolean') {
      if (typeof raw.decision !== 'boolean') {
        throw invalid();
      }
      answers[id] = {
        type: 'boolean',
        decision: raw.decision,
        probability: null,
      };
      continue;
    }
    if (
      question.type !== 'choice' ||
      typeof raw.choice !== 'string' ||
      !Object.hasOwn(question.criteria, raw.choice)
    ) {
      throw invalid();
    }
    answers[id] = {
      type: 'choice',
      choice: raw.choice,
      confidence: null,
      probabilities: null,
    };
  }
  return answers;
}

function questionsForChat(
  questions: ClassificationRequest['questions'],
  provider: string,
  maxQuestions: number
): Array<[string, ClassificationQuestion]> {
  const entries = validateClassificationQuestions(questions, provider);
  let choices = 0;
  for (const [, question] of entries) {
    if (question.type === 'score') {
      throw new ClassificationError(
        'unsupported_question',
        'chat classification cannot score expected values',
        { provider }
      );
    }
    if (question.type === 'choice') {
      choices += Object.keys(question.criteria).length;
    }
  }
  if (entries.length > maxQuestions || choices > MAX_CHOICE_OPTIONS) {
    throw new ClassificationError(
      'unsupported_question',
      'classifier question batch too large',
      { provider }
    );
  }
  return entries;
}

/** Strict provider schema or strict tool calling, without an agent loop or fabricated probabilities. */
export function createStructuredChatClassifier(
  options: StructuredChatClassifierOptions
): Classifier {
  const provider = options.providerId ?? PROVIDER_ID;
  if (!['jsonSchema', 'functionCalling'].includes(options.method)) {
    throw new ClassificationError(
      'unsupported_mode',
      'unsupported structured classifier mode',
      {
        provider,
      }
    );
  }
  if (!options.modelId.trim()) {
    throw new ClassificationError(
      'bad_request',
      'classifier requires a model id',
      {
        provider,
      }
    );
  }
  const maxQuestions = options.maxQuestions ?? MAX_QUESTIONS;
  if (
    !Number.isSafeInteger(maxQuestions) ||
    maxQuestions < 1 ||
    maxQuestions > MAX_QUESTIONS
  ) {
    throw new ClassificationError(
      'bad_request',
      'invalid classifier question limit',
      {
        provider,
      }
    );
  }
  const { model, method } = options;
  const modelId = options.modelId.trim();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const onAnswered = options.onAnswered;

  return {
    id: provider,
    model: modelId,
    async classify(
      request: ClassificationRequest
    ): Promise<ClassificationResult> {
      const started = performance.now();
      return withClassificationDeadline(
        provider,
        request.timeoutMs ?? timeoutMs,
        request.signal,
        async (signal, waitFor) => {
          const entries = questionsForChat(
            request.questions,
            provider,
            maxQuestions
          );
          let input: string;
          try {
            input =
              CLASSIFICATION_PROMPT_PREFIX +
              JSON.stringify({
                state: request.state,
                questions: Object.fromEntries(entries),
              });
          } catch {
            throw new ClassificationError(
              'bad_request',
              'invalid classifier request',
              {
                provider,
              }
            );
          }
          let structured: ReturnType<typeof model.withStructuredOutput>;
          try {
            structured = model.withStructuredOutput(decisionSchema(entries), {
              name: 'ClassifyDecisions',
              method,
              strict: true,
              includeRaw: true,
            });
          } catch {
            throw new ClassificationError(
              'unsupported_mode',
              'model cannot enforce the requested strict mode',
              {
                provider,
              }
            );
          }
          await waitFor(Promise.resolve());
          let output: Awaited<ReturnType<typeof structured.invoke>>;
          try {
            output = await waitFor(
              structured.invoke(
                [
                  new SystemMessage(
                    'Classify the state using the questions. Return only the required decisions. Do not estimate probabilities, confidence, or token usage.'
                  ),
                  new HumanMessage(input),
                ],
                { signal }
              )
            );
          } catch (error) {
            if (error instanceof ClassificationError) {
              throw error;
            }
            throw providerFailure(error, provider);
          }
          const answers = readDecisions(output.parsed, entries, provider);
          const result: ClassificationResult = {
            model: modelId,
            answers,
            usage: readChatUsage(output.raw),
          };
          await waitFor(Promise.resolve());
          try {
            onAnswered?.(
              request.label ?? 'classify',
              performance.now() - started
            );
          } catch {
            // A callback error must not make a valid classification fail.
          }
          return result;
        }
      );
    },
  };
}

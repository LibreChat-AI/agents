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
import { withClassificationDeadline } from './deadline';

const MAX_QUESTIONS = 32;
const MAX_CHOICE_OPTIONS = 128;
const DEFAULT_TIMEOUT_MS = 20_000;
const PROVIDER_ID = 'structured-chat';

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
    properties[id] =
      question.type === 'boolean'
        ? {
          type: 'object',
          properties: { decision: { type: 'boolean' } },
          required: ['decision'],
          additionalProperties: false,
        }
        : {
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
      const entries = validateClassificationQuestions(
        request.questions,
        provider
      );
      let choices = 0;
      for (const [, question] of entries) {
        if (question.type === 'score') {
          throw new ClassificationError(
            'unsupported_question',
            'chat classification cannot score expected values',
            {
              provider,
            }
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
          {
            provider,
          }
        );
      }

      return withClassificationDeadline(
        provider,
        request.timeoutMs ?? timeoutMs,
        request.signal,
        async (signal, waitFor) => {
          let input: string;
          try {
            input = JSON.stringify({
              state: request.state,
              questions: request.questions,
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
            throw new ClassificationError(
              'network',
              'structured classifier request failed',
              {
                provider,
              }
            );
          }
          const answers = readDecisions(output.parsed, entries, provider);
          const rawUsage = isClassificationObject(output.raw)
            ? output.raw.usage_metadata
            : null;
          const result: ClassificationResult = {
            model: modelId,
            answers,
            usage: readClassificationUsage(rawUsage),
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

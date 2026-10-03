import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type {
  DecisionModel,
  DecisionQuestion,
  DecisionResult,
  DecisionRequest,
} from './types';
import type { DecisionOutputMethod } from './structuredOutput';
import { DecisionError, isDecisionObject, readDecisionUsage } from './types';
import { withDecisionStructuredOutput } from './structuredOutput';
import { failureForStatus, retryAfterMs } from './transport';
import { validateDecisionQuestions } from './questions';
import { DECISION_PROMPT_PREFIX } from './traceMarker';
import { withDecisionDeadline } from './deadline';

const MAX_QUESTIONS = 32;
const MAX_CHOICE_OPTIONS = 128;
const DEFAULT_TIMEOUT_MS = 20_000;
const PROVIDER_ID = 'structured-chat';

export function readChatUsage(raw: unknown): DecisionResult['usage'] {
  if (!isDecisionObject(raw)) {
    return null;
  }
  const usage = readDecisionUsage(raw.usage_metadata);
  const response = isDecisionObject(raw.response_metadata)
    ? raw.response_metadata
    : null;
  const nested = isDecisionObject(response?.metadata)
    ? response.metadata
    : null;
  const bedrock = isDecisionObject(nested?.usage)
    ? nested.usage
    : response?.usage;
  if (
    usage?.inputTokens === undefined ||
    !isDecisionObject(bedrock) ||
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

function providerFailure(error: unknown, provider: string): DecisionError {
  const details = isDecisionObject(error) ? error : null;
  const metadata = isDecisionObject(details?.$metadata)
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
  const failure = new DecisionError(
    status == null ? 'network' : failureForStatus(status),
    'structured decision model request failed',
    { provider, status }
  );
  const headers = details?.headers;
  if (headers instanceof Headers) {
    failure.retryAfterMs = retryAfterMs(headers.get('retry-after'));
  }
  return failure;
}

export interface StructuredChatDecisionModelOptions {
  /** An already configured chat model whose chosen method enforces strict schemas. */
  model: BaseChatModel;
  modelId: string;
  providerId?: string;
  /** Verified OpenAI/Azure strict modes or Anthropic strict tools; other paths fail closed. */
  method: DecisionOutputMethod;
  timeoutMs?: number;
  maxQuestions?: number;
  onAnswered?: (label: string, ms: number) => void;
}

function decisionSchema(
  entries: Array<[string, DecisionQuestion]>
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
  entries: Array<[string, DecisionQuestion]>,
  provider: string
): DecisionResult['answers'] {
  const invalid = (): DecisionError =>
    new DecisionError(
      'malformed_response',
      'invalid structured decision model answer',
      {
        provider,
      }
    );
  if (
    !isDecisionObject(parsed) ||
    Object.keys(parsed).length !== 1 ||
    !isDecisionObject(parsed.answers) ||
    Object.keys(parsed.answers).length !== entries.length
  ) {
    throw invalid();
  }
  const answers: DecisionResult['answers'] = Object.create(null);
  for (const [id, question] of entries) {
    if (!Object.hasOwn(parsed.answers, id)) {
      throw invalid();
    }
    const raw = parsed.answers[id];
    if (!isDecisionObject(raw) || Object.keys(raw).length !== 1) {
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
  questions: DecisionRequest['questions'],
  provider: string,
  maxQuestions: number
): Array<[string, DecisionQuestion]> {
  const entries = validateDecisionQuestions(questions, provider);
  let choices = 0;
  for (const [, question] of entries) {
    if (question.type === 'score') {
      throw new DecisionError(
        'unsupported_question',
        'chat decision cannot score expected values',
        { provider }
      );
    }
    if (question.type === 'choice') {
      choices += Object.keys(question.criteria).length;
    }
  }
  if (entries.length > maxQuestions || choices > MAX_CHOICE_OPTIONS) {
    throw new DecisionError(
      'unsupported_question',
      'decision model question batch too large',
      { provider }
    );
  }
  return entries;
}

/** Strict provider schema or strict tool calling, without an agent loop or fabricated probabilities. */
export function createStructuredChatDecisionModel(
  options: StructuredChatDecisionModelOptions
): DecisionModel {
  const provider = options.providerId ?? PROVIDER_ID;
  if (!['jsonSchema', 'functionCalling'].includes(options.method)) {
    throw new DecisionError(
      'unsupported_mode',
      'unsupported structured decision model mode',
      {
        provider,
      }
    );
  }
  if (!options.modelId.trim()) {
    throw new DecisionError(
      'bad_request',
      'decision model requires a model id',
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
    throw new DecisionError(
      'bad_request',
      'invalid decision model question limit',
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
    async decide(request: DecisionRequest): Promise<DecisionResult> {
      const started = performance.now();
      return withDecisionDeadline(
        provider,
        request.timeoutMs ?? timeoutMs,
        request.signal,
        async (signal, waitFor) => {
          let entries: Array<[string, DecisionQuestion]>;
          let input: string;
          try {
            entries = questionsForChat(
              request.questions,
              provider,
              maxQuestions
            );
            input =
              DECISION_PROMPT_PREFIX +
              JSON.stringify({
                state: request.state,
                questions: Object.fromEntries(entries),
              });
          } catch (error) {
            if (error instanceof DecisionError) {
              throw error;
            }
            throw new DecisionError(
              'bad_request',
              'invalid decision model request',
              {
                provider,
              }
            );
          }
          let structured: ReturnType<typeof model.withStructuredOutput>;
          try {
            structured = withDecisionStructuredOutput(
              model,
              decisionSchema(entries),
              method
            );
          } catch {
            throw new DecisionError(
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
            if (error instanceof DecisionError) {
              throw error;
            }
            throw providerFailure(error, provider);
          }
          const answers = readDecisions(output.parsed, entries, provider);
          const result: DecisionResult = {
            model: modelId,
            answers,
            usage: readChatUsage(output.raw),
          };
          await waitFor(Promise.resolve());
          try {
            onAnswered?.(
              request.label ?? 'decide',
              performance.now() - started
            );
          } catch {
            // A callback error must not make a valid decision fail.
          }
          return result;
        }
      );
    },
  };
}

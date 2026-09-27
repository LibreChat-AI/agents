/** Typed decisions over content, independent of the model that makes them. */

export type ClassificationJson =
  | string
  | number
  | boolean
  | null
  | ClassificationJson[]
  | { [key: string]: ClassificationJson };

export type ClassificationText =
  | string
  | ClassificationJson[]
  | { [key: string]: ClassificationJson };

export type ClassificationState = ClassificationText;

export interface BooleanCriteria {
  true?: ClassificationText;
  false?: ClassificationText;
}

/** A string criterion describes the yes side of a yes/no question. */
export interface BooleanQuestion {
  type: 'boolean';
  instructions: ClassificationText;
  criteria?: BooleanCriteria | string;
}

export interface ChoiceQuestion {
  type: 'choice';
  instructions: ClassificationText;
  criteria: Record<string, ClassificationText | null>;
}

/** The measured answer is an expected level, which may fall between two rubric levels. */
export interface ScoreQuestion {
  type: 'score';
  instructions: ClassificationText;
  criteria: ClassificationText[];
}

export type ClassificationQuestion =
  | BooleanQuestion
  | ChoiceQuestion
  | ScoreQuestion;

/** `null` is an unmeasured decision, not a probability of zero. */
export type BooleanAnswer =
  | { type: 'boolean'; probability: number; decision?: never }
  | { type: 'boolean'; probability: null; decision: boolean };

export interface ChoiceAnswer {
  type: 'choice';
  choice: string;
  /** Backend-specific concentration measure; Jev and Laya use different definitions. */
  confidence: number | null;
  /** `null` when no distribution was measured; never substitute an empty distribution. */
  probabilities: Record<string, number> | null;
}

export interface ScoreAnswer {
  type: 'score';
  /** Expected value over rubric levels, not an LLM's selection of one level. */
  score: number;
  /** Backend-specific concentration measure; do not transfer thresholds between checkpoints. */
  confidence: number | null;
  probabilities: Record<string, number> | null;
}

export type ClassificationAnswer = BooleanAnswer | ChoiceAnswer | ScoreAnswer;

/** Unknown counts are omitted, never inferred from an answer or replaced by zero. */
export interface ClassificationUsage {
  inputTokens?: number;
  outputTokens?: number;
}

export interface ClassificationRequest {
  state: ClassificationState;
  questions: Record<string, ClassificationQuestion>;
  signal?: AbortSignal;
  label?: string;
  /** Overrides the provider's timeout for this request alone. */
  timeoutMs?: number;
}

export interface ClassificationResult {
  model: string;
  /** A requested answer can be omitted; callers must handle `undefined` explicitly. */
  answers: Record<string, ClassificationAnswer | undefined>;
  usage: ClassificationUsage | null;
}

export interface Classifier {
  readonly id: string;
  readonly model: string;
  classify(request: ClassificationRequest): Promise<ClassificationResult>;
}

export type ClassificationDialect = 'port' | 'systemone';

/** A bearer key, or a per-request token minter; a 401 requests one fresh token for that call. */
export type ClassificationCredential =
  | string
  | ((options: { refresh: boolean; signal?: AbortSignal }) => Promise<string>);

export interface ClassificationProviderSettings {
  /** Full URL of the classify endpoint, not a base path. */
  baseURL?: string;
  model?: string;
  dialect?: ClassificationDialect;
  requestKey?: string;
  responseKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  /** Auth is required by default; self-hosted Laya can be unauthenticated. */
  requiresAuth?: boolean;
  apiKeyEnv?: string;
}

export type ClassificationFailure =
  | 'timeout'
  | 'aborted'
  | 'rate_limited'
  | 'unauthorized'
  | 'bad_request'
  | 'server_error'
  | 'network'
  | 'unsupported_question'
  | 'unsupported_mode'
  | 'malformed_response';

export class ClassificationError extends Error {
  readonly failure: ClassificationFailure;
  readonly provider: string;
  readonly status?: number;
  retryAfterMs?: number;

  constructor(
    failure: ClassificationFailure,
    message: string,
    options?: { provider?: string; status?: number }
  ) {
    super(message);
    this.name = 'ClassificationError';
    this.failure = failure;
    this.provider = options?.provider ?? 'unknown';
    this.status = options?.status;
  }
}

/** A JSON boundary guard; never trust a parsed response's declared TypeScript type. */
export function isClassificationObject(
  value: unknown
): value is { [key: string]: unknown } {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function readClassificationUsage(
  raw: unknown
): ClassificationUsage | null {
  if (!isClassificationObject(raw)) {
    return null;
  }
  const input = raw.input_tokens;
  const output = raw.output_tokens;
  const usage: ClassificationUsage = {};
  if (typeof input === 'number' && Number.isSafeInteger(input) && input >= 0) {
    usage.inputTokens = input;
  }
  if (
    typeof output === 'number' &&
    Number.isSafeInteger(output) &&
    output >= 0
  ) {
    usage.outputTokens = output;
  }
  return Object.keys(usage).length > 0 ? usage : null;
}

export function isBooleanAnswer(
  answer: ClassificationAnswer | undefined
): answer is BooleanAnswer {
  return answer?.type === 'boolean';
}

export function isChoiceAnswer(
  answer: ClassificationAnswer | undefined
): answer is ChoiceAnswer {
  return answer?.type === 'choice';
}

export function isScoreAnswer(
  answer: ClassificationAnswer | undefined
): answer is ScoreAnswer {
  return answer?.type === 'score';
}

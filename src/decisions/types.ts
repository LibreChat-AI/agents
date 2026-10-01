/** Typed decisions over content, independent of the model that makes them. */

export type DecisionJson =
  | string
  | number
  | boolean
  | null
  | DecisionJson[]
  | { [key: string]: DecisionJson };

export type DecisionText =
  string | DecisionJson[] | { [key: string]: DecisionJson };

export type DecisionState = DecisionText;

export interface BooleanCriteria {
  true?: DecisionText;
  false?: DecisionText;
}

/** A string criterion describes the yes side of a yes/no question. */
export interface BooleanQuestion {
  type: 'boolean';
  instructions: DecisionText;
  criteria?: BooleanCriteria | string;
}

export interface ChoiceQuestion {
  type: 'choice';
  instructions: DecisionText;
  criteria: Record<string, DecisionText | null>;
}

/** The measured answer is an expected level, which may fall between two rubric levels. */
export interface ScoreQuestion {
  type: 'score';
  instructions: DecisionText;
  criteria: DecisionText[];
}

export type DecisionQuestion = BooleanQuestion | ChoiceQuestion | ScoreQuestion;

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

export type DecisionAnswer = BooleanAnswer | ChoiceAnswer | ScoreAnswer;

/** Unknown counts are omitted, never inferred from an answer or replaced by zero. */
export interface DecisionUsage {
  inputTokens?: number;
  outputTokens?: number;
}

export interface DecisionRequest {
  state: DecisionState;
  questions: Record<string, DecisionQuestion>;
  signal?: AbortSignal;
  label?: string;
  /** Overrides the provider's timeout for this request alone. */
  timeoutMs?: number;
}

export interface DecisionResult {
  model: string;
  /** A requested answer can be omitted; callers must handle `undefined` explicitly. */
  answers: Record<string, DecisionAnswer | undefined>;
  usage: DecisionUsage | null;
}

export interface DecisionModel {
  readonly id: string;
  readonly model: string;
  decide(request: DecisionRequest): Promise<DecisionResult>;
}

export type DecisionDialect = 'port' | 'systemone';

/** A bearer key, or a per-request token minter; a 401 requests one fresh token for that call. */
export type DecisionCredential =
  | string
  | ((options: { refresh: boolean; signal?: AbortSignal }) => Promise<string>);

export interface DecisionProviderSettings {
  /** Full URL of the decision endpoint, not a base path. */
  baseURL?: string;
  model?: string;
  dialect?: DecisionDialect;
  requestKey?: string;
  responseKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  /** Auth is required by default; self-hosted Laya can be unauthenticated. */
  requiresAuth?: boolean;
  apiKeyEnv?: string;
}

export type DecisionFailure =
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

export class DecisionError extends Error {
  readonly failure: DecisionFailure;
  readonly provider: string;
  readonly status?: number;
  retryAfterMs?: number;

  constructor(
    failure: DecisionFailure,
    message: string,
    options?: { provider?: string; status?: number }
  ) {
    super(message);
    this.name = 'DecisionError';
    this.failure = failure;
    this.provider = options?.provider ?? 'unknown';
    this.status = options?.status;
  }
}

/** A JSON boundary guard; never trust a parsed response's declared TypeScript type. */
export function isDecisionObject(
  value: unknown
): value is { [key: string]: unknown } {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function readDecisionUsage(raw: unknown): DecisionUsage | null {
  if (!isDecisionObject(raw)) {
    return null;
  }
  const input = raw.input_tokens;
  const output = raw.output_tokens;
  const usage: DecisionUsage = {};
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
  answer: DecisionAnswer | undefined
): answer is BooleanAnswer {
  return answer?.type === 'boolean';
}

export function isChoiceAnswer(
  answer: DecisionAnswer | undefined
): answer is ChoiceAnswer {
  return answer?.type === 'choice';
}

export function isScoreAnswer(
  answer: DecisionAnswer | undefined
): answer is ScoreAnswer {
  return answer?.type === 'score';
}

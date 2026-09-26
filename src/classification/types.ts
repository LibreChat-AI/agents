/**
 * The classification port: typed questions about a piece of content, answered by a classifier
 * with a probability per answer. A System One host (TypeSafe's Jev, the same model through a
 * gateway) answers all of them in one call; the port is what a consumer codes against, so a
 * second host is a preset, not a branch.
 */

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

/** A yes/no question, answered with the probability of yes. A string criterion describes the yes side. */
export interface BooleanQuestion {
  type: 'boolean';
  instructions: ClassificationText;
  criteria?: BooleanCriteria | string;
}

/** Pick one of the named options; a `null` description means the name speaks for itself. */
export interface ChoiceQuestion {
  type: 'choice';
  instructions: ClassificationText;
  criteria: Record<string, ClassificationText | null>;
}

/** Rate against an ordered rubric; the answer is the expected level, which may fall between two. */
export interface ScoreQuestion {
  type: 'score';
  instructions: ClassificationText;
  criteria: ClassificationText[];
}

export type ClassificationQuestion =
  | BooleanQuestion
  | ChoiceQuestion
  | ScoreQuestion;

export interface BooleanAnswer {
  type: 'boolean';
  probability: number;
}

export interface ChoiceAnswer {
  type: 'choice';
  choice: string;
  /** `null` when the provider cannot measure it, which is not the same as 0. */
  confidence: number | null;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type: 'score';
  score: number;
  confidence: number | null;
  /** Probability per rubric level, keyed by the level's index as a string. */
  probabilities: Record<string, number>;
}

export type ClassificationAnswer = BooleanAnswer | ChoiceAnswer | ScoreAnswer;

export interface ClassificationUsage {
  inputTokens: number;
  outputTokens: number;
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
  answers: Record<string, ClassificationAnswer>;
  usage: ClassificationUsage;
}

export interface Classifier {
  readonly id: string;
  readonly model: string;
  classify(request: ClassificationRequest): Promise<ClassificationResult>;
}

/** Which wire vocabulary a host speaks: the port's own, or System One's (`noul` for boolean). */
export type ClassificationDialect = 'port' | 'systemone';

/**
 * A bearer credential: the key itself, or a function that mints one. The function form serves
 * hosts whose tokens expire (the ClickHouse gateway's hourly Okta token): the transport calls it
 * before each request and once more with `refresh: true` after a 401, then retries that request.
 */
export type ClassificationCredential =
  | string
  | ((options: { refresh: boolean }) => Promise<string>);

/** A host, as settings rather than code. */
export interface ClassificationProviderSettings {
  /** Full URL of the classify endpoint, not a base path. */
  baseURL?: string;
  model?: string;
  dialect?: ClassificationDialect;
  /** Nests `state` and `questions` under this key, for hosts that wrap them. */
  requestKey?: string;
  /** Reads the answer envelope from this key, for hosts that wrap the response. */
  responseKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  /** The environment variable an operator puts the key in. */
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

import type {
  DecisionAnswer,
  DecisionDialect,
  DecisionQuestion,
} from './types';
import { isDecisionObject } from './types';

/**
 * Hosts in front of System One (ClickHouse gateway, OpenRouter) round every reported number to
 * two decimals, so each value can sit up to half a hundredth from the measured one, and never
 * outside `[0, 1]`. A `systemone` answer whose probabilities and score all sit on that grid is read
 * under this rounding model: its distribution must be one that some measured distribution summing
 * to exactly one rounds to, and its score must round from the expectation of such a distribution.
 * The model is chosen once per answer, so a single off-grid number puts the whole answer, score
 * and distribution alike, under the four-decimal bounds, as it does every `port` answer. `FLOAT_SLACK` absorbs
 * binary floating-point error at the exact bounds; anything further out is an inconsistent answer.
 */
const REPORTED_HALF_STEP = 0.005;
const REPORTED_STEPS = Math.round(1 / (2 * REPORTED_HALF_STEP));
const FLOAT_SLACK = 1e-9;

/** Whether two-decimal rounding could have produced this number. */
function isTwoDecimal(value: number): boolean {
  const steps = value * REPORTED_STEPS;
  return Math.abs(steps - Math.round(steps)) <= FLOAT_SLACK * REPORTED_STEPS;
}

/** The measured value a rounded one can come from: within a half-step, and inside `[0, 1]`. */
function measuredRange(value: number): [number, number] {
  return [
    Math.max(0, value - REPORTED_HALF_STEP),
    Math.min(1, value + REPORTED_HALF_STEP),
  ];
}

/** Whether a measured distribution summing to exactly one could round to these values. */
function roundsFromUnitSum(values: readonly number[]): boolean {
  let lowest = 0;
  let highest = 0;
  for (const value of values) {
    const [low, high] = measuredRange(value);
    lowest += low;
    highest += high;
  }
  return lowest <= 1 + FLOAT_SLACK && highest >= 1 - FLOAT_SLACK;
}

/** The four-decimal bound on how far a distribution may sum from one. */
function sumsToOne(values: readonly number[]): boolean {
  const total = values.reduce((sum, value) => sum + value, 0);
  const tolerance = Math.min(
    0.02,
    Math.max(0.01, values.length * 0.00005 + 0.001)
  );
  return Math.abs(total - 1) <= tolerance;
}

/**
 * The lowest and highest expectation of a measured distribution summing to one that rounds to
 * `byLevel` (indexed by level): every level starts at its lowest measured value, and the mass left
 * to reach one fills the lowest levels first for the minimum and the highest first for the maximum.
 */
function expectationRange(byLevel: readonly number[]): [number, number] {
  const ranges = byLevel.map(measuredRange);
  let base = 0;
  let spare = 1;
  for (const [level, [low]] of ranges.entries()) {
    base += level * low;
    spare -= low;
  }
  const fill = (levels: readonly number[]): number => {
    let expectation = base;
    let left = Math.max(0, spare);
    for (const level of levels) {
      if (left <= 0) {
        break;
      }
      const [low, high] = ranges[level];
      const added = Math.min(high - low, left);
      expectation += level * added;
      left -= added;
    }
    return expectation;
  };
  const ascending = ranges.map((_, level) => level);
  return [fill(ascending), fill([...ascending].reverse())];
}

/** Whether a two-decimal score rounds from the expectation of a distribution behind `byLevel`. */
function roundsFromExpectation(
  score: number,
  byLevel: readonly number[]
): boolean {
  const [lowest, highest] = expectationRange(byLevel);
  return (
    score >= lowest - REPORTED_HALF_STEP - FLOAT_SLACK &&
    score <= highest + REPORTED_HALF_STEP + FLOAT_SLACK
  );
}

/** The four-decimal bound on how far a score may sit from its distribution's expectation. */
function nearExpectation(score: number, byLevel: readonly number[]): boolean {
  const levels = byLevel.length;
  const expectation = byLevel.reduce(
    (sum, probability, level) => sum + level * probability,
    0
  );
  const tolerance = Math.min(
    0.02,
    0.00005 * (1 + (levels * (levels - 1)) / 2) + 0.001
  );
  return (
    Number.isFinite(expectation) && Math.abs(score - expectation) <= tolerance
  );
}

interface WireQuestion {
  type: string;
  instructions: DecisionQuestion['instructions'];
  criteria?: DecisionQuestion['criteria'];
}

interface WireAnswer {
  type?: unknown;
  probability?: unknown;
  noul?: unknown;
  choice?: unknown;
  score?: unknown;
  confidence?: unknown;
  probabilities?: unknown;
}

/** System One calls a yes/no question a `noul`; both dialects use a `{true, false}` criterion. */
export function toWireQuestion(
  question: DecisionQuestion,
  dialect: DecisionDialect
): WireQuestion {
  const type =
    dialect === 'systemone' && question.type === 'boolean'
      ? 'noul'
      : question.type;
  if (question.criteria == null) {
    return { type, instructions: question.instructions };
  }
  const criteria =
    question.type === 'boolean' && typeof question.criteria === 'string'
      ? { true: question.criteria }
      : question.criteria;
  return { type, instructions: question.instructions, criteria };
}

function isProbability(value: unknown): value is number {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
  );
}

/**
 * `undefined` means malformed; `null` means no distribution was reported. Whether the values sum
 * to one depends on the rounding model of the whole answer, so `readAnswer` checks that.
 */
function probabilitiesOf(
  raw: unknown,
  answerType: 'choice' | 'score',
  question?: DecisionQuestion,
  selectedChoice?: string
): Record<string, number> | null | undefined {
  if (raw == null) {
    return null;
  }
  if (!isDecisionObject(raw)) {
    return undefined;
  }
  const entries = Object.entries(raw);
  if (entries.length === 0) {
    return null;
  }
  if (
    (question?.type === 'choice' &&
      entries.length !== Object.keys(question.criteria).length) ||
    (question?.type === 'score' && entries.length !== question.criteria.length)
  ) {
    return undefined;
  }
  const probabilities: Record<string, number> = Object.create(null);
  let maximum = 0;
  for (const [key, value] of entries) {
    if (!isProbability(value)) {
      return undefined;
    }
    if (question?.type === 'choice' && !Object.hasOwn(question.criteria, key)) {
      return undefined;
    }
    if (
      answerType === 'score' &&
      (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= entries.length)
    ) {
      return undefined;
    }
    probabilities[key] = value;
    maximum = Math.max(maximum, value);
  }
  if (
    selectedChoice !== undefined &&
    (!Object.hasOwn(probabilities, selectedChoice) ||
      probabilities[selectedChoice] < maximum)
  ) {
    return undefined;
  }
  return probabilities;
}

/** A validated answer, or `null` for one that cannot safely be read. */
export function readAnswer(
  answer: unknown,
  dialect: DecisionDialect,
  question?: DecisionQuestion
): DecisionAnswer | null {
  if (!isDecisionObject(answer)) {
    return null;
  }
  const record: WireAnswer = answer;
  const booleanType = dialect === 'systemone' ? 'noul' : 'boolean';
  const probability =
    dialect === 'systemone' ? record.noul : record.probability;
  if (record.type === booleanType) {
    if (
      !isProbability(probability) ||
      (question != null && question.type !== 'boolean')
    ) {
      return null;
    }
    return { type: 'boolean', probability };
  }
  if (record.type !== 'choice' && record.type !== 'score') {
    return null;
  }
  const confidence = record.confidence;
  if (confidence != null && !isProbability(confidence)) {
    return null;
  }
  const probabilities = probabilitiesOf(
    record.probabilities,
    record.type,
    question,
    record.type === 'choice' && typeof record.choice === 'string'
      ? record.choice
      : undefined
  );
  if (probabilities === undefined) {
    return null;
  }
  const values = probabilities === null ? [] : Object.values(probabilities);
  /** One rounding model for the answer: its score and its distribution are read under it together. */
  const rounded =
    dialect === 'systemone' &&
    values.every(isTwoDecimal) &&
    (record.type !== 'score' ||
      (typeof record.score === 'number' && isTwoDecimal(record.score)));
  if (
    probabilities !== null &&
    !(rounded ? roundsFromUnitSum(values) : sumsToOne(values))
  ) {
    return null;
  }
  if (record.type === 'choice') {
    if (
      typeof record.choice !== 'string' ||
      (question != null &&
        (question.type !== 'choice' ||
          !Object.hasOwn(question.criteria, record.choice)))
    ) {
      return null;
    }
    return {
      type: 'choice',
      choice: record.choice,
      confidence: confidence ?? null,
      probabilities,
    };
  }
  if (
    typeof record.score !== 'number' ||
    !Number.isFinite(record.score) ||
    record.score < 0 ||
    (question != null &&
      (question.type !== 'score' ||
        record.score > question.criteria.length - 1))
  ) {
    return null;
  }
  if (probabilities !== null) {
    /** Score keys are the consecutive levels `0..n-1`, checked by `probabilitiesOf`. */
    const byLevel: number[] = [];
    for (const [level, probability] of Object.entries(probabilities)) {
      byLevel[Number(level)] = probability;
    }
    const consistent = rounded
      ? roundsFromExpectation(record.score, byLevel)
      : nearExpectation(record.score, byLevel);
    if (!consistent) {
      return null;
    }
  }
  return {
    type: 'score',
    score: record.score,
    confidence: confidence ?? null,
    probabilities,
  };
}

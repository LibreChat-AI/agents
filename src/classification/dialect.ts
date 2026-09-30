import type {
  ClassificationAnswer,
  ClassificationDialect,
  ClassificationQuestion,
} from './types';
import { isClassificationObject } from './types';

interface WireQuestion {
  type: string;
  instructions: ClassificationQuestion['instructions'];
  criteria?: ClassificationQuestion['criteria'];
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
  question: ClassificationQuestion,
  dialect: ClassificationDialect
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

/** `undefined` means malformed; `null` means no distribution was reported. */
function probabilitiesOf(
  raw: unknown,
  question?: ClassificationQuestion,
  selectedChoice?: string
): Record<string, number> | null | undefined {
  if (raw == null) {
    return null;
  }
  if (!isClassificationObject(raw)) {
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
  let total = 0;
  let maximum = 0;
  for (const [key, value] of entries) {
    if (!isProbability(value)) {
      return undefined;
    }
    if (question?.type === 'choice' && !Object.hasOwn(question.criteria, key)) {
      return undefined;
    }
    if (
      question?.type === 'score' &&
      (!/^(0|[1-9][0-9]*)$/.test(key) ||
        Number(key) >= question.criteria.length)
    ) {
      return undefined;
    }
    probabilities[key] = value;
    total += value;
    maximum = Math.max(maximum, value);
  }
  if (
    selectedChoice !== undefined &&
    (!Object.hasOwn(probabilities, selectedChoice) ||
      probabilities[selectedChoice] < maximum)
  ) {
    return undefined;
  }
  const roundingTolerance = Math.min(
    0.02,
    Math.max(0.01, entries.length * 0.00005 + 0.001)
  );
  return Math.abs(total - 1) <= roundingTolerance ? probabilities : undefined;
}

/** A validated answer, or `null` for one that cannot safely be read. */
export function readAnswer(
  answer: unknown,
  dialect: ClassificationDialect,
  question?: ClassificationQuestion
): ClassificationAnswer | null {
  if (!isClassificationObject(answer)) {
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
    question,
    record.type === 'choice' && typeof record.choice === 'string'
      ? record.choice
      : undefined
  );
  if (probabilities === undefined) {
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
    let weightedScore = 0;
    let levels = 0;
    for (const [level, probability] of Object.entries(probabilities)) {
      weightedScore += Number(level) * probability;
      levels += 1;
    }
    const roundingTolerance = Math.min(
      0.02,
      0.00005 * (1 + (levels * (levels - 1)) / 2) + 0.001
    );
    if (Math.abs(record.score - weightedScore) > roundingTolerance) {
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

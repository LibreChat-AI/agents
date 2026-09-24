import type {
  ClassificationAnswer,
  ClassificationDialect,
  ClassificationQuestion,
} from './types';

interface WireQuestion {
  type: string;
  instructions: unknown;
  criteria?: unknown;
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

/**
 * System One calls a yes/no question a `noul` and wants its criteria as a `{true, false}` pair;
 * the port calls it a boolean and lets a caller describe only the yes side with a string.
 */
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

function probabilitiesOf(record: WireAnswer): Record<string, number> {
  if (
    record.probabilities == null ||
    typeof record.probabilities !== 'object'
  ) {
    return {};
  }
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(
    record.probabilities as Record<string, unknown>
  )) {
    if (typeof value === 'number') {
      out[key] = value;
    }
  }
  return out;
}

/** An answer in the port's shape, or `null` for one that cannot be read — never an invented one. */
export function readAnswer(
  answer: unknown,
  dialect: ClassificationDialect
): ClassificationAnswer | null {
  if (answer == null || typeof answer !== 'object') {
    return null;
  }
  const record = answer as WireAnswer;
  const probabilities = probabilitiesOf(record);
  const confidence =
    typeof record.confidence === 'number' ? record.confidence : null;

  const booleanType = dialect === 'systemone' ? 'noul' : 'boolean';
  const probability =
    dialect === 'systemone' ? record.noul : record.probability;
  if (record.type === booleanType && typeof probability === 'number') {
    return { type: 'boolean', probability };
  }
  if (record.type === 'choice' && typeof record.choice === 'string') {
    return { type: 'choice', choice: record.choice, confidence, probabilities };
  }
  if (record.type === 'score' && typeof record.score === 'number') {
    return { type: 'score', score: record.score, confidence, probabilities };
  }
  return null;
}

import type {
  BooleanCriteria,
  BooleanQuestion,
  ChoiceQuestion,
  ScoreQuestion,
  ClassificationText,
  ClassificationQuestion,
} from './types';
import { ClassificationError, isClassificationObject } from './types';

function validQuestionId(id: string): boolean {
  if (id.length === 0 || id.length > 128) {
    return false;
  }
  for (let index = 0; index < id.length; index++) {
    if (id.charCodeAt(index) < 32) {
      return false;
    }
  }
  return true;
}

function validBooleanCriteria(criteria: BooleanQuestion['criteria']): boolean {
  if (criteria === undefined || typeof criteria === 'string') {
    return true;
  }
  if (!isClassificationObject(criteria)) {
    return false;
  }
  for (const [side, description] of Object.entries(criteria)) {
    if (
      (side !== 'true' && side !== 'false') ||
      (description !== undefined &&
        typeof description !== 'string' &&
        !Array.isArray(description) &&
        !isClassificationObject(description))
    ) {
      return false;
    }
  }
  return true;
}

export function validateClassificationQuestions(
  questions: Record<string, ClassificationQuestion>,
  provider: string
): Array<[string, ClassificationQuestion]> {
  if (!isClassificationObject(questions)) {
    throw new ClassificationError(
      'bad_request',
      'invalid classifier questions',
      {
        provider,
      }
    );
  }
  const supplied = Object.entries(questions);
  if (supplied.length === 0) {
    throw new ClassificationError(
      'bad_request',
      'classifier requires questions',
      {
        provider,
      }
    );
  }
  const entries: Array<[string, ClassificationQuestion]> = [];
  for (const [id, question] of supplied) {
    if (
      !validQuestionId(id) ||
      !isClassificationObject(question) ||
      !['boolean', 'choice', 'score'].includes(question.type) ||
      (question.type === 'boolean' &&
        !validBooleanCriteria(question.criteria)) ||
      (question.type === 'choice' &&
        (!isClassificationObject(question.criteria) ||
          Object.keys(question.criteria).length === 0)) ||
      (question.type === 'score' &&
        (!Array.isArray(question.criteria) || question.criteria.length < 2))
    ) {
      throw new ClassificationError(
        'bad_request',
        'invalid classifier question',
        {
          provider,
        }
      );
    }
    if (question.type === 'choice') {
      entries.push([id, { ...question, criteria: { ...question.criteria } }]);
    } else if (question.type === 'score') {
      entries.push([id, { ...question, criteria: [...question.criteria] }]);
    } else {
      entries.push([
        id,
        {
          ...question,
          ...(isClassificationObject(question.criteria)
            ? { criteria: { ...question.criteria } }
            : {}),
        },
      ]);
    }
  }
  return entries;
}

/** A yes/no question; `criteria` describes what counts as yes (a string) or both sides. */
export function booleanQuestion(
  instructions: ClassificationText,
  criteria?: BooleanCriteria | string
): BooleanQuestion {
  return criteria == null
    ? { type: 'boolean', instructions }
    : { type: 'boolean', instructions, criteria };
}

/** Pick one of the named options; a `null` description means the name speaks for itself. */
export function choiceQuestion(
  instructions: ClassificationText,
  criteria: Record<string, ClassificationText | null>
): ChoiceQuestion {
  return { type: 'choice', instructions, criteria };
}

/** Rate against an ordered rubric, level 0 first. */
export function scoreQuestion(
  instructions: ClassificationText,
  levels: ClassificationText[]
): ScoreQuestion {
  return { type: 'score', instructions, criteria: levels };
}

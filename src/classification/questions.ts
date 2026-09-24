import type {
  BooleanCriteria,
  BooleanQuestion,
  ChoiceQuestion,
  ScoreQuestion,
  ClassificationText,
} from './types';

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

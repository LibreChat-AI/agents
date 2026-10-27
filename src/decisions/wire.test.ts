import { readFileSync } from 'fs';
import type { DecisionQuestion } from './index';
import { parseEnvelope, readAnswer } from './index';

interface WireRecord {
  state: string;
  model: string;
  answers: Record<string, Record<string, unknown>>;
  usage: Record<string, number>;
}

interface WireFixture {
  questions: Record<
    string,
    { type: string; instructions: string; criteria?: unknown }
  >;
  records: WireRecord[];
}

/** Real jev-1.13.0 answers recorded through the ClickHouse gateway on 2026-10-18 (synthetic code states). */
const fixture = JSON.parse(
  readFileSync('src/decisions/fixtures/jev-1.13.0.json', 'utf-8')
) as WireFixture;

/** The fixture's questions in SDK form; System One's `noul` is the SDK's boolean. */
const questions: Record<string, DecisionQuestion> = Object.fromEntries(
  Object.entries(fixture.questions).map(([id, question]) => [
    id,
    JSON.parse(
      JSON.stringify({
        ...question,
        type: question.type === 'noul' ? 'boolean' : question.type,
      })
    ) as DecisionQuestion,
  ])
);

/** The bounds this module used before it accounted for hosts rounding to two decimals. */
function rejectedByFourDecimalBounds(answer: Record<string, unknown>): boolean {
  if (answer.type === 'noul') {
    return false;
  }
  const probabilities = Object.entries(
    answer.probabilities as Record<string, number>
  );
  const n = probabilities.length;
  const total = probabilities.reduce((sum, [, p]) => sum + p, 0);
  if (
    Math.abs(total - 1) > Math.min(0.02, Math.max(0.01, n * 0.00005 + 0.001))
  ) {
    return true;
  }
  if (answer.type !== 'score') {
    return false;
  }
  const expected = probabilities.reduce(
    (sum, [level, p]) => sum + Number(level) * p,
    0
  );
  return (
    Math.abs((answer.score as number) - expected) >
    Math.min(0.02, 0.00005 * (1 + (n * (n - 1)) / 2) + 0.001)
  );
}

describe('System One wire fidelity', () => {
  it('records noul, multi-level score, and multi-option choice answers', () => {
    const types = new Map<string, number>();
    for (const record of fixture.records) {
      expect(record.model).toBe('jev-1.13.0');
      for (const answer of Object.values(record.answers)) {
        types.set(
          String(answer.type),
          (types.get(String(answer.type)) ?? 0) + 1
        );
      }
    }
    expect(fixture.records).toHaveLength(40);
    expect(Object.fromEntries(types)).toEqual({
      noul: 80,
      score: 120,
      choice: 80,
    });
    const kind = fixture.questions.kind.criteria as Record<string, string>;
    expect(Object.keys(kind)).toHaveLength(5);
  });

  it('reads every recorded gateway answer, which the four-decimal bounds mostly rejected', () => {
    let answersRejectedBefore = 0;
    let callsRejectedBefore = 0;
    for (const record of fixture.records) {
      const before = Object.values(record.answers).filter(
        rejectedByFourDecimalBounds
      ).length;
      answersRejectedBefore += before;
      callsRejectedBefore += before > 0 ? 1 : 0;

      for (const [id, answer] of Object.entries(record.answers)) {
        expect(readAnswer(answer, 'systemone', questions[id])).not.toBeNull();
      }
      const result = parseEnvelope(
        JSON.stringify(record),
        'systemone',
        (answer, question) => readAnswer(answer, 'systemone', question),
        undefined,
        questions
      );
      expect(Object.keys(result.answers)).toHaveLength(7);
      expect(result.usage?.inputTokens).toBeGreaterThan(0);
    }
    /** All 43 were score answers (of 120); one bad answer fails the whole call, so 30 of 40 calls threw. */
    expect(answersRejectedBefore).toBe(43);
    expect(callsRejectedBefore).toBe(30);
  });

  it('accepts a score only where the expectation of its distribution can round to it', () => {
    const score = questions.readability;
    const probabilities = { '0': 0, '1': 0, '2': 0.5, '3': 0.5 };
    // Measured values summing to one put the expectation in [2.48, 2.505], so it rounds to 2.48
    // through 2.51; a bound from the level count alone would also allow 2.47 and 2.52. 2.475 is
    // no two-decimal number, so it gets the four-decimal bound around 2.5.
    for (const value of [2.48, 2.49, 2.5, 2.51]) {
      expect(
        readAnswer(
          { type: 'score', score: value, probabilities },
          'systemone',
          score
        )
      ).toMatchObject({ type: 'score', score: value });
    }
    for (const value of [2.465, 2.47, 2.475, 2.52, 2.535, 3]) {
      expect(
        readAnswer(
          { type: 'score', score: value, probabilities },
          'systemone',
          score
        )
      ).toBeNull();
    }
    const nines = Object.fromEntries(
      Array.from({ length: 10 }, (_, level) => [
        String(level),
        level === 9 ? 1 : 0,
      ])
    );
    // Level nine holds at least 0.995, so the expectation is at least 8.955.
    expect(
      readAnswer(
        { type: 'score', score: 8.95, probabilities: nines },
        'systemone'
      )
    ).toMatchObject({ type: 'score', score: 8.95 });
    expect(
      readAnswer(
        { type: 'score', score: 8.94, probabilities: nines },
        'systemone'
      )
    ).toBeNull();
  });

  it('keeps the four-decimal bounds for the port dialect and for unrounded numbers', () => {
    const spread = (length: number, value: number): Record<string, number> =>
      Object.fromEntries(
        Array.from({ length }, (_, index) => [`k${index}`, value])
      );
    const uniform = { '0': 0.25, '1': 0.25, '2': 0.25, '3': 0.25 };
    for (const dialect of ['port', 'systemone'] as const) {
      // 0.015 is no two-decimal rounding, so 100 of them summing to 1.5 is never a rounded one.
      expect(
        readAnswer(
          { type: 'choice', choice: 'k0', probabilities: spread(100, 0.015) },
          dialect
        )
      ).toBeNull();
      expect(
        readAnswer(
          { type: 'choice', choice: 'k0', probabilities: spread(10, 0.105) },
          dialect
        )
      ).toBeNull();
      expect(
        readAnswer(
          { type: 'score', score: 1.53, probabilities: uniform },
          dialect
        )
      ).toBeNull();
      // Unrounded numbers that agree to four decimals read under either dialect.
      expect(
        readAnswer(
          {
            type: 'score',
            score: 1.3766,
            probabilities: { '0': 0.1234, '1': 0.3766, '2': 0.5 },
          },
          dialect
        )
      ).toMatchObject({ type: 'score', score: 1.3766 });
    }
    // Two-decimal rounding drift is a System One host's, not a generic HTTP host's.
    const rounded = {
      type: 'choice',
      choice: 'k0',
      probabilities: { ...spread(5, 0.2), k0: 0.22 },
    };
    expect(readAnswer(rounded, 'systemone')).toMatchObject({ choice: 'k0' });
    expect(readAnswer(rounded, 'port')).toBeNull();
    expect(
      readAnswer(
        { type: 'score', score: 1.52, probabilities: uniform },
        'systemone'
      )
    ).toMatchObject({ score: 1.52 });
    expect(
      readAnswer({ type: 'score', score: 1.52, probabilities: uniform }, 'port')
    ).toBeNull();
  });

  it('reads a score and its distribution under one rounding model', () => {
    // On the grid, but summing to 1.02: only a rounding of some true distribution.
    const drifted = { '0': 0.25, '1': 0.25, '2': 0.25, '3': 0.27 };
    expect(
      readAnswer(
        { type: 'score', score: 1.53, probabilities: drifted },
        'systemone'
      )
    ).toMatchObject({ score: 1.53 });
    // These scores are no rounding, so the distribution must sum to one within four decimals as
    // well, even though each is within the four-decimal bound of the reported expectation 1.56.
    for (const score of [1.5599, 1.5601]) {
      expect(
        readAnswer(
          { type: 'score', score, probabilities: drifted },
          'systemone'
        )
      ).toBeNull();
    }
    // An unrounded score keeps working with a distribution that sums to one.
    expect(
      readAnswer(
        {
          type: 'score',
          score: 1.5001,
          probabilities: { '0': 0.25, '1': 0.25, '2': 0.25, '3': 0.25 },
        },
        'systemone'
      )
    ).toMatchObject({ score: 1.5001 });
  });

  it('accepts a distribution whose rounding drift scales with its size, and nothing further', () => {
    const kind = questions.kind;
    // True 0.196 x4 + 0.216 rounds to 0.20 x4 + 0.22: a sum of 1.02, inside 0.005 * 5.
    expect(
      readAnswer(
        {
          type: 'choice',
          choice: 'config',
          probabilities: {
            util: 0.2,
            hook: 0.2,
            route: 0.2,
            test: 0.2,
            config: 0.22,
          },
        },
        'systemone',
        kind
      )
    ).toMatchObject({ type: 'choice', choice: 'config' });
    expect(
      readAnswer(
        {
          type: 'choice',
          choice: 'config',
          probabilities: {
            util: 0.2,
            hook: 0.2,
            route: 0.2,
            test: 0.2,
            config: 0.23,
          },
        },
        'systemone',
        kind
      )
    ).toBeNull();
    // Three options allow 0.015, so a sum of 1.02 is inconsistent rather than rounded.
    expect(
      readAnswer(
        {
          type: 'choice',
          choice: 'ts',
          probabilities: { ts: 0.34, py: 0.34, go: 0.34 },
        },
        'systemone',
        questions.language
      )
    ).toBeNull();
    expect(
      readAnswer(
        {
          type: 'score',
          score: 1,
          probabilities: { '0': 0.34, '1': 0.34, '2': 0.34 },
        },
        'systemone',
        questions.risk
      )
    ).toBeNull();
  });

  it('caps rounding drift at what values summing to one can produce, however many there are', () => {
    const zeros = (length: number): Record<string, number> =>
      Object.fromEntries(
        Array.from({ length }, (_, index) => [`k${index}`, 0])
      );
    // 250 half-steps would allow a drift of 1.25, but two certainties each need at least 0.995.
    const twoCertainties = { ...zeros(250), k0: 1, k1: 1 };
    expect(
      readAnswer(
        { type: 'choice', choice: 'k0', probabilities: twoCertainties },
        'systemone'
      )
    ).toBeNull();
    // A zero hides at most a half-step, so 0.5 and 0.52 among zeros cannot come from one.
    expect(
      readAnswer(
        {
          type: 'choice',
          choice: 'b',
          probabilities: { a: 0.5, b: 0.52, c: 0, d: 0 },
        },
        'systemone'
      )
    ).toBeNull();
    const spread = Object.fromEntries(
      Array.from({ length: 250 }, (_, index) => [
        `k${index}`,
        index < 200 ? 0.01 : 0,
      ])
    );
    // Two hundred true values of 0.005 round up to 0.01 each: a sum of two, still a rounding.
    expect(
      readAnswer(
        { type: 'choice', choice: 'k0', probabilities: spread },
        'systemone'
      )
    ).toMatchObject({ type: 'choice', choice: 'k0' });

    const levels = Object.fromEntries(
      Array.from({ length: 300 }, (_, level) => [String(level), 0])
    );
    levels['0'] = 1;
    // Level zero holds at least 0.995, so the expectation is at most 1.495 and rounds to 1.5 at most.
    expect(
      readAnswer(
        { type: 'score', score: 1.49, probabilities: levels },
        'systemone'
      )
    ).toMatchObject({ type: 'score', score: 1.49 });
    for (const score of [1.51, 2, 210]) {
      expect(
        readAnswer({ type: 'score', score, probabilities: levels }, 'systemone')
      ).toBeNull();
    }
  });
});

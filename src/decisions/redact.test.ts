import { redact, redactExcerpt } from './redact';

describe('redact cost', () => {
  it('keeps memory proportional to the text when a short secret repeats throughout it', () => {
    const text = '}'.repeat(1 << 20);
    const secrets = Array.from({ length: 200 }, () => '}');
    expect(redact(text, secrets)).toBe('[REDACTED]');
  });

  it('merges overlapping occurrences of one secret into a single run', () => {
    expect(redact('xababay', ['aba'])).toBe('x[REDACTED]y');
    expect(redact('ab ab', ['ab'])).toBe('[REDACTED] [REDACTED]');
  });

  it('finds JWT shapes in time linear in the text', () => {
    const text = 'eyJ-'.repeat(200_000);
    const started = performance.now();
    expect(redact(text)).toBe('[REDACTED]');
    expect(performance.now() - started).toBeLessThan(500);
  });

  it('recognizes the same JWT shapes a backtracking pattern would', () => {
    const segment = 'abcdefgh';
    const jwt = `eyJ${segment}.${segment}.${segment}`;
    expect(redact(`a ${jwt} b`)).toBe('a [REDACTED] b');
    expect(redact(`x-${jwt}.tail`)).toBe('x-[REDACTED].tail');
    expect(redact(`eyJ-eyJ-${jwt}`)).toBe('[REDACTED]');
    expect(redact(`eyJ${segment}-${jwt}`)).toBe('[REDACTED]');
    expect(redact(`eyJ1234567.${segment}.${segment}`)).toBe(
      `eyJ1234567.${segment}.${segment}`
    );
    expect(redact(`eyJ${segment}.${segment}.1234567`)).toBe(
      `eyJ${segment}.${segment}.1234567`
    );
    expect(redact(`eyJ${segment}.${segment} ${segment}`)).toBe(
      `eyJ${segment}.${segment} ${segment}`
    );
    expect(redact(`${jwt}.${jwt}`)).toBe('[REDACTED].[REDACTED]');
    expect(redact(`seyJ${segment}.${segment}.${segment}`)).toBe(
      `seyJ${segment}.${segment}.${segment}`
    );
  });

  it('matches the backtracking JWT pattern on generated text', () => {
    const backtracking = /\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}/g;
    const pieces = ['eyJ', 'abcd', 'ab', '-', '.', ' ', 'x', 'e'];
    let seed = 7;
    const next = (): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed;
    };
    for (let round = 0; round < 5000; round++) {
      let text = '';
      while (text.length < 30 + (next() % 10)) {
        text += pieces[next() % pieces.length];
      }
      /** Shorter than an opaque token, so only the JWT shape can match. */
      text = text.slice(0, 39);
      expect(redact(text)).toBe(text.replace(backtracking, '[REDACTED]'));
    }
  });

  it('still throws for a secrets list it cannot iterate', () => {
    const unreadable = null as unknown as string[];
    expect(() => redact('text', unreadable)).toThrow(TypeError);
  });
});

describe('redactExcerpt', () => {
  it('collapses whitespace and keeps the first characters', () => {
    expect(
      redactExcerpt('  bad\n\n  password\thunter22 \n', ['hunter22'], 200)
    ).toBe('bad password [REDACTED]');
    expect(redactExcerpt('abcdef', [], 3)).toBe('abc');
  });

  it('redacts a listed secret whose whitespace the excerpt collapses', () => {
    expect(redactExcerpt('x abc\tdef y', ['abc\tdef'], 200)).toBe(
      'x [REDACTED] y'
    );
    expect(redactExcerpt('x abc\n  def y', ['abc def'], 200)).toBe(
      'x [REDACTED] y'
    );
  });

  it('stays bounded against many distinct secrets that match everywhere', () => {
    const text = 'a'.repeat(900_000);
    const secrets = Array.from({ length: 600 }, (_, i) => 'a'.repeat(i + 1));
    const started = performance.now();
    expect(redactExcerpt(text, secrets, 200)).toBe('[REDACTED]');
    expect(performance.now() - started).toBeLessThan(2000);
  });

  it('hides the first characters of a listed secret the window cuts off', () => {
    const text = `${'A'.repeat(4090)} hunter2xyz and more`;
    const result = redactExcerpt(text, ['hunter2xyz'], 200);
    expect(result).toBe('[REDACTED]');
    expect(result).not.toContain('hunte');
  });

  it('hides the first characters of an unlisted token the window cuts off', () => {
    const text = `${'A'.repeat(4060)} ${'B'.repeat(100)} tail`;
    const result = redactExcerpt(text, [], 200);
    expect(result).toBe('[REDACTED] [REDACTED]');
    expect(result).not.toContain('B');
  });

  it('reads a bounded prefix of a long text with whitespace to collapse', () => {
    const text = 'a\t'.repeat(10_000_000);
    const started = performance.now();
    expect(redactExcerpt(text, [], 200)).toBe('a '.repeat(100));
    expect(performance.now() - started).toBeLessThan(100);
  });

  it('hides a listed secret that the raw prefix cuts through', () => {
    const text = `x${' '.repeat(16_380)}hunter22 tail`;
    const result = redactExcerpt(text, ['hunter22'], 200);
    expect(result).toBe('[REDACTED]');
    expect(result).not.toContain('hun');
  });

  it('leaves the tail alone when the text fits the window', () => {
    expect(redactExcerpt(`${'A'.repeat(50)} short`, ['zzzzzzzzzz'], 200)).toBe(
      '[REDACTED] short'
    );
  });
});

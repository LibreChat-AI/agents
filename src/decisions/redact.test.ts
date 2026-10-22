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

  it('leaves the tail alone when the text fits the window', () => {
    expect(redactExcerpt(`${'A'.repeat(50)} short`, ['zzzzzzzzzz'], 200)).toBe(
      '[REDACTED] short'
    );
  });
});

import { describe, it, expect } from '@jest/globals';
import {
  HARD_MAX_TOOL_RESULT_CHARS,
  HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE,
  calculateMaxToolResultChars,
  calculateMaxTotalToolOutputSize,
  sliceWithoutSplittingSurrogates,
} from '@/utils/truncation';

describe('truncation helpers', () => {
  describe('sliceWithoutSplittingSurrogates', () => {
    it.each([
      [0, 1, 'a'],
      [0, 2, 'a'],
      [0, 3, 'a🧠'],
      [1, 3, '🧠'],
      [1, 2, ''],
      [2, 3, ''],
      [2, 4, 'b'],
      [2, 6, 'b🍱'],
      [2, 5, 'b'],
      [3, 3, ''],
      [5, 2, ''],
      [-2, undefined, 'c'],
      [-3, undefined, '🍱c'],
      [0, 100, 'a🧠b🍱c'],
    ])('slices [%i, %s) without splitting a pair', (start, end, expected) => {
      expect(sliceWithoutSplittingSurrogates('a🧠b🍱c', start, end)).toBe(
        expected
      );
    });

    it('leaves BMP text and intact compound emoji unchanged', () => {
      const text = 'ASCII 漢字 café ✈️ 👩🏽‍💻';
      expect(sliceWithoutSplittingSurrogates(text, 0)).toBe(text);
      expect(sliceWithoutSplittingSurrogates(text, 0, 5)).toBe('ASCII');
      expect(sliceWithoutSplittingSurrogates('', 0, 1)).toBe('');
    });

    it('drops orphaned edges in already-clipped collector buffers', () => {
      expect(sliceWithoutSplittingSurrogates('a\ud83e', 0)).toBe('a');
      expect(sliceWithoutSplittingSurrogates('\udde0b', 0)).toBe('b');
    });
  });

  describe('calculateMaxToolResultChars', () => {
    it('returns the hard cap when context tokens are missing', () => {
      expect(calculateMaxToolResultChars()).toBe(HARD_MAX_TOOL_RESULT_CHARS);
      expect(calculateMaxToolResultChars(undefined)).toBe(
        HARD_MAX_TOOL_RESULT_CHARS
      );
      expect(calculateMaxToolResultChars(0)).toBe(HARD_MAX_TOOL_RESULT_CHARS);
      expect(calculateMaxToolResultChars(-100)).toBe(
        HARD_MAX_TOOL_RESULT_CHARS
      );
    });

    it('computes 30% of context-window characters for normal inputs', () => {
      // 100k tokens * 0.3 = 30k tokens * 4 chars/token = 120k chars
      expect(calculateMaxToolResultChars(100_000)).toBe(120_000);
    });

    it('clamps to the hard cap for large context windows', () => {
      // 1M tokens * 0.3 * 4 = 1.2M chars, exceeds 400k cap
      expect(calculateMaxToolResultChars(1_000_000)).toBe(
        HARD_MAX_TOOL_RESULT_CHARS
      );
    });
  });

  describe('calculateMaxTotalToolOutputSize', () => {
    it('returns the absolute hard cap when no per-output is provided', () => {
      expect(calculateMaxTotalToolOutputSize()).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
      expect(calculateMaxTotalToolOutputSize(0)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
      expect(calculateMaxTotalToolOutputSize(-1)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
    });

    it('doubles the per-output cap by default', () => {
      expect(calculateMaxTotalToolOutputSize(100_000)).toBe(200_000);
      expect(calculateMaxTotalToolOutputSize(1)).toBe(2);
    });

    it('clamps the doubled value to HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE', () => {
      // 4M * 2 = 8M, exceeds 5M
      expect(calculateMaxTotalToolOutputSize(4_000_000)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
      // Right at the boundary: 2.5M * 2 = 5M (no clamp).
      expect(calculateMaxTotalToolOutputSize(2_500_000)).toBe(5_000_000);
      // Just past it: 2_500_001 * 2 = 5_000_002 -> clamped.
      expect(calculateMaxTotalToolOutputSize(2_500_001)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
    });
  });
});

import { describe, expect, it } from '@jest/globals';
import {
  appendArtifactTruncationWarning,
  normalizeArtifactTruncation,
} from '../ArtifactTruncation';

const marker = {
  code: 'artifact_truncated',
  reasons: { max_files: 70 },
  skipped: ['report_070.csv', 'report_068.csv'],
  skipped_count: 70,
};

describe('artifact truncation', () => {
  it('normalizes the Code API marker without echoing unexpected fields', () => {
    expect(
      normalizeArtifactTruncation({ ...marker, detail: 'private data' })
    ).toEqual(marker);
  });

  it.each([
    null,
    [],
    { ...marker, code: 'not_truncated' },
    { ...marker, skipped_count: -1 },
    { ...marker, skipped_count: 0, skipped: [] },
    { ...marker, skipped_count: 1 },
    { ...marker, skipped_count: 1.5 },
    { ...marker, skipped_count: '70' },
    { ...marker, reasons: null },
    { ...marker, reasons: {} },
    { ...marker, reasons: { max_files: 0 } },
    { ...marker, reasons: { max_files: 69 } },
    { ...marker, reasons: { max_files: 71 } },
    { ...marker, reasons: { max_files: 35, size: 36 } },
    { ...marker, reasons: { max_files: 35, size: 34 } },
    {
      ...marker,
      skipped: ['report.csv'],
      skipped_count: 1,
    },
    {
      ...marker,
      reasons: { max_files: Number.MAX_SAFE_INTEGER + 1 },
      skipped_count: Number.MAX_SAFE_INTEGER + 1,
    },
    {
      ...marker,
      reasons: { max_files: Number.MAX_SAFE_INTEGER, size: 1 },
      skipped_count: Number.MAX_SAFE_INTEGER,
    },
    { ...marker, skipped_count: NaN },
    { ...marker, reasons: { max_files: Infinity } },
    { ...marker, reasons: { max_files: 1, unexpected: 1 } },
    { ...marker, reasons: { size: -1 } },
    { ...marker, reasons: { path: 1.2 } },
    { ...marker, skipped: [null] },
    {
      ...marker,
      skipped: Array.from({ length: 21 }, (_, index) => `report_${index}.csv`),
    },
  ])('rejects a malformed or oversized marker: %j', (value) => {
    expect(normalizeArtifactTruncation(value)).toBeUndefined();
  });

  it('accepts omission counts that match or exceed the reported paths', () => {
    const fullyReported = {
      ...marker,
      reasons: { max_files: 2 },
      skipped_count: 2,
    };
    expect(normalizeArtifactTruncation(fullyReported)).toEqual(fullyReported);
    expect(normalizeArtifactTruncation({ ...marker, skipped: [] })).toEqual({
      ...marker,
      skipped: [],
    });
  });

  it.each([
    { max_files: 14, depth: 14, size: 14, path: 14, unreadable: 14 },
    { max_files: 70, size: 0 },
  ])('accepts consistent reason totals: %j', (reasons) => {
    const value = { ...marker, reasons };
    expect(normalizeArtifactTruncation(value)).toEqual(value);
  });

  it('accepts the largest exactly representable omission count', () => {
    const value = {
      ...marker,
      reasons: { max_files: Number.MAX_SAFE_INTEGER - 1, size: 1 },
      skipped_count: Number.MAX_SAFE_INTEGER,
    };
    expect(normalizeArtifactTruncation(value)).toEqual(value);
  });

  it('copies metadata without sharing mutable state across responses', () => {
    const value = {
      ...marker,
      reasons: { ...marker.reasons },
      skipped: [...marker.skipped],
    };
    const first = normalizeArtifactTruncation(value);
    const second = normalizeArtifactTruncation(value);
    value.reasons.max_files = 1;
    value.skipped.push('later.csv');

    expect(first).toEqual(marker);
    expect(second).toEqual(marker);
    expect(first?.reasons).not.toBe(second?.reasons);
    expect(first?.skipped).not.toBe(second?.skipped);
  });

  it('reports omissions and bounds the displayed paths without suggesting a rerun', () => {
    const truncation = normalizeArtifactTruncation(marker);
    const output = appendArtifactTruncationWarning(
      'stdout:\ndone\n',
      truncation
    );

    expect(output).toContain('70 file(s) were omitted from delivery');
    expect(output).toContain('(max_files: 70)');
    expect(output).toContain('report_070.csv, report_068.csv (2 of 70 shown)');
    expect(output).toContain('do not rerun automatically');
    expect(appendArtifactTruncationWarning('done', undefined)).toBe('done');
  });
});

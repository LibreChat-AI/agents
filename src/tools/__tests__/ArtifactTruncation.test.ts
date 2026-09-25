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
    expect(normalizeArtifactTruncation({ ...marker, skipped_count: 2 })).toEqual({
      ...marker,
      skipped_count: 2,
    });
    expect(normalizeArtifactTruncation({ ...marker, skipped: [] })).toEqual({
      ...marker,
      skipped: [],
    });
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

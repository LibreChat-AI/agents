import type { ArtifactTruncation, ArtifactTruncationReason } from '@/types';

const MAX_REPORTED_TRUNCATED_PATHS = 20;
const ARTIFACT_TRUNCATION_REASONS = new Set<string>([
  'max_files',
  'depth',
  'size',
  'path',
  'unreadable',
]);

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

export function normalizeArtifactTruncation(
  value: unknown
): ArtifactTruncation | undefined {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }

  const candidate = value as Partial<ArtifactTruncation>;
  if (
    candidate.code !== 'artifact_truncated' ||
    !isNonNegativeInteger(candidate.skipped_count) ||
    candidate.skipped_count === 0 ||
    candidate.reasons == null ||
    typeof candidate.reasons !== 'object' ||
    Array.isArray(candidate.reasons) ||
    !Array.isArray(candidate.skipped) ||
    candidate.skipped.length > MAX_REPORTED_TRUNCATED_PATHS ||
    candidate.skipped.length > candidate.skipped_count ||
    candidate.skipped.some((path) => typeof path !== 'string')
  ) {
    return undefined;
  }

  const reasons: ArtifactTruncation['reasons'] = {};
  for (const [reason, count] of Object.entries(candidate.reasons)) {
    if (
      !ARTIFACT_TRUNCATION_REASONS.has(reason) ||
      !isNonNegativeInteger(count)
    ) {
      return undefined;
    }
    reasons[reason as ArtifactTruncationReason] = count;
  }

  return {
    code: candidate.code,
    reasons,
    skipped: [...candidate.skipped],
    skipped_count: candidate.skipped_count,
  };
}

export function appendArtifactTruncationWarning(
  output: string,
  truncation: ArtifactTruncation | undefined
): string {
  if (truncation == null) {
    return output;
  }

  const reasons = Object.entries(truncation.reasons)
    .map(([reason, count]) => `${reason}: ${count}`)
    .join(', ');
  const shown = truncation.skipped.length;
  const paths =
    shown > 0
      ? ` Not delivered: ${truncation.skipped.join(', ')}${shown < truncation.skipped_count ? ` (${shown} of ${truncation.skipped_count} shown)` : ''}.`
      : '';
  const warning = `Note: ${truncation.skipped_count} file(s) were omitted from delivery${reasons ? ` (${reasons})` : ''}.${paths} Write fewer files per execution or combine them into an archive. The code itself ran; do not rerun automatically because it may have had side effects.`;
  return `${output.trimEnd()}\n${warning}\n`;
}

const REDACTED = '[REDACTED]';
const BEARER = /\b(bearer)\s+[^\s"',;]+/gi;
const JWT = /\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}/g;
/** Long opaque runs (API keys, session tokens); also catches 40-hex digests, which is acceptable in a log line. */
const OPAQUE = /[A-Za-z0-9_-]{40,}/g;

/**
 * Removes credentials from text bound for a log line or a status row: every
 * listed secret verbatim, then anything shaped like a bearer value, a JWT, or
 * a long opaque token. Unlisted short secrets are not recognizable by shape,
 * so pass every credential the text could contain.
 */
export function redact(
  text: string,
  secrets: ReadonlyArray<string | null | undefined> = []
): string {
  let result = String(text);
  const listed = secrets
    .map((secret) => (typeof secret === 'string' ? secret.trim() : ''))
    .filter((secret) => secret !== '')
    .sort((a, b) => b.length - a.length);
  for (const secret of listed) {
    result = result.split(secret).join(REDACTED);
  }
  return result
    .replace(BEARER, `$1 ${REDACTED}`)
    .replace(JWT, REDACTED)
    .replace(OPAQUE, REDACTED);
}

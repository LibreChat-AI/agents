/** Explicitly empty credentials disable environment fallback. */
export function resolveKagiKey(apiKey?: string): string {
  const key = apiKey ?? process.env.KAGI_API_KEY;
  if (key == null || key.trim() === '') {
    throw new Error('Kagi API key is required');
  }
  return key;
}

/** Reject credentials and non-web schemes before exposing URLs to a host or provider.
 * Ignore fragments for extraction matching, but preserve paths, queries and schemes. */
export function getKagiUrl(url: string): string | undefined {
  try {
    const parsed = new URL(url);
    if (
      (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') ||
      parsed.username !== '' ||
      parsed.password !== ''
    ) {
      return;
    }
    parsed.hash = '';
    return parsed.href;
  } catch {
    return;
  }
}

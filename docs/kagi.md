# Kagi Search and Extract

Kagi is a native search provider and an independently selectable scraper in
`createSearchTool`. Both use the [official v1 OpenAPI](https://kagi.com/api/docs/_spec/openapi.yaml?download):

- Search: `POST https://kagi.com/api/v1/search`
- Extract: `POST https://kagi.com/api/v1/extract`
- Authentication: `Authorization: Bearer <API key>`

These are not the legacy GET search endpoint or its `Bot` authentication scheme.
Endpoints are fixed; no custom base URL is accepted, and redirects are disabled.

```typescript
import { createSearchTool } from '@librechat/agents';

const webSearch = createSearchTool({
  searchProvider: 'kagi',
  scraperProvider: 'kagi',
  kagiApiKey: process.env.KAGI_API_KEY,
  kagiSearchOptions: { limit: 8, timeout: 10000 },
  kagiScraperOptions: { timeout: 7500 },
  rerankerType: 'none', // Or configure any existing reranker.
});

await webSearch.invoke({
  query: 'latest TypeScript releases',
  country: 'us',
  images: true,
  videos: true,
  news: true,
});
```

## Configuration

| Field                        | Behavior                                                                                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kagiApiKey`                 | Shared search/extract credential; falls back to `KAGI_API_KEY` only when omitted/nullish. A missing, blank or explicitly empty key is a configuration error.                                                                       |
| `kagiSearchOptions.limit`    | Maximum results per workflow; defaults to the requested result count (8 in the tool). Finite values are truncated and clamped to 1–1024; non-finite values use 8. This limits returned results, not Kagi's underlying search work. |
| `kagiSearchOptions.timeout`  | HTTP timeout in **milliseconds**, default 10000. Not sent as Kagi's server-side `timeout` (which is measured in seconds).                                                                                                          |
| `kagiScraperOptions.apiKey`  | Optional Extract-only override; otherwise uses `kagiApiKey`, then the environment. Explicit `''` never falls back.                                                                                                                 |
| `kagiScraperOptions.timeout` | HTTP timeout in **milliseconds**, default 7500. Global `scraperTimeout` takes precedence.                                                                                                                                          |
| `httpAgent`, `httpsAgent`    | Global injected transport agents take precedence over agents in either Kagi options bag.                                                                                                                                           |

Search and scraping are independent: Kagi search can use Serper, Firecrawl, Tavily,
CRW or Keenable extraction, and any other search provider can use Kagi Extract.
Unselected Kagi providers do not require a key. Existing defaults are unchanged.
No dependencies or package configuration changes are needed.

## Search mapping

Requests contain `{ query, workflow, format: 'json', limit }`. The main query uses
`workflow: 'search'`; requested image, video and news sub-searches use `images`,
`videos` and `news`, respectively. JSON response buckets `search`, `image`, `video`
and `news` map into the existing organic/media/news structures. Titles, URLs,
snippets, timestamps and associated images are preserved; `related_search` titles
become related queries. Image results without a separate image use their resource
URL. An explicit empty result array is a successful zero-result query; missing or
malformed expected buckets are failures, not fabricated empty successes.

- Country maps to `filters.region` in uppercase (`uk` is normalized to `GB`).
- Day/week/month use `lens.time_relative`. Kagi has no hourly interval, so
  `PAST_HOUR` is deliberately broadened to `day` rather than sending an unsupported value.
- Past year uses `filters.after`, an ISO UTC date one calendar year ago. Kagi's
  date filters have day precision, not an exact rolling timestamp.
- The current OpenAPI **does** support `safe_search`: level 0 maps to `false`,
  levels 1 and 2 to `true`. There is no separate moderate/strict distinction.
  Direct API calls that omit safe search leave Kagi's documented default (`true`)
  intact; the tool passes its existing default level 1.

## Extract mapping and failures

Extract posts `{ pages: [{ url }], format: 'json' }` in batches of at most ten
pages. Empty input makes no request. HTTP(S) URLs are forwarded to Kagi for remote
extraction; LibreChat agents never fetch those pages directly. The endpoint's
summary allows HTTP(S), although its `pageInput` description specifically requires
HTTPS; Kagi may reject HTTP pages. Invalid/non-HTTP(S) URLs fail locally.

Markdown is preserved verbatim by the scraper, then passes through the existing
content normalization, chunking, reranking, citations, output-budget and metrics
pipeline. Results are matched to requested URLs, not response order. Missing
pages, absent/null/empty Markdown and per-page errors fail only those pages;
successful pages remain usable, including when the response has plural `errors`
diagnostics. A singular envelope `error` fails the batch, even if data is present.
Batches are independent, so a failed batch does not discard another's success.

API/network failures return the existing `success: false` contract with stable,
provider-specific messages. Raw exceptions, API error text/codes, response bodies
and authorization headers are never forwarded to logs, callbacks, artifacts or
tool output. No provider-specific spans or changes to Langfuse shaping are added;
the existing `web_search` tool observation and redaction policy remain in place.

## Privacy, billing and host integration

Kagi receives search queries/filters and the page URLs sent for extraction. Search
and Extract are separately billed APIs; requesting additional workflows or
extracting pages can incur additional charges. See [Kagi pricing](https://kagi.com/api/pricing).
No live API credentials are needed for the focused HTTP-spy tests.

This implementation is limited to `librechat-agents`. Host configuration/API/UI
wiring, any dual-context Dockerfile, and installing a built local tarball into
LibreChat are separate host-side work. After `npm run build`, a host integrator can
package with `npm pack --ignore-scripts` and install that tarball using their
existing no-save/no-lock workflow. This change does not modify LibreChat, install
into its workspace, or alter dependency manifests/lockfiles.

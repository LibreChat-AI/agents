# Reranking

`Reranker.rerank({ query, documents, topK?, signal?, timeoutMs? })` returns ranked input indices and provider-specific scores. Indices preserve candidate identity even when two documents have identical text. A successful ranking is ordered by descending score. Scores are not portable confidence thresholds. Missing usage is `null`.

## Light entry point

`@librechat/agents/rerank` exports the `Reranker` types, the never-throwing outcome layer, `createSystemOneReranker`, `createCrossEncoderReranker`, `createBm25Reranker`, `noneReranker`, `withFallback`, and `probePin` without loading LangChain, any other third-party package, or a Node built-in; pair it with `@librechat/agents/decisions` to build the decision model. `createWebSearchReranker` extends the search tool's reranker, so it is exported only from the root entry.

## Host-owned web-search ranking

`createSearchTool({ reranker })` now honors an injected `SearchReranker` before constructing its configured built-in reranker. Hosts can supply a plain `{ provider, rerank(query, documents, topK?, metrics?) }` object. Extending `BaseReranker` is not required. Existing Jina, Cohere, rag_api, Infinity, and `none` selection stays unchanged when there is no injection.

For an index-based `Reranker`, `createWebSearchReranker(reranker, { timeoutMs?, logger? })` maps indices to highlights, validates ordering/ranges/duplicates, and records exactly one observation per attempt. Invalid or failed rankings fall back to input order with neutral scores. Error metrics never copy provider exception text. Empty inputs and zero result limits require no provider work. Without an explicit adapter `timeoutMs`, the wrapper retains its own 10-second outer bound but does not override the injected ranker's configured deadline. An explicit adapter timeout is forwarded as a request override.

## Rerank outcomes

`Reranker.rerank` rejects on failure; that contract and `createWebSearchReranker` are unchanged. An `OutcomeReranker` instead resolves every call to a `RerankOutcome` and never throws or rejects:

```ts
interface RerankOutcome {
  results: { index: number; score: number | null; kept?: boolean }[]; // a full permutation
  provider: 'systemone' | 'cross-encoder' | 'bm25' | 'none';
  model: string; // what the host echoed
  status: 'ok' | 'degraded' | 'skipped';
  reason: string | null; // null when ok; otherwise the first failed step's reason
  attempts: RerankAttempt[]; // every step that ran or was skipped, with its own usage
  durationMs: number;
  usage: RerankUsage; // summed over attempts, estimates included
  pinVerified?: boolean;
  threshold?: number;
}
```

`results` always holds every input index once: score descending, ties in input order, `null` scores last. A `skipped` outcome is identity order with every score `null`, so a caller can always serve the order it already had. There is no `topK`; the caller slices. A request with a blank query, a document that is not a string (holes included), a `deadlineMs` that is not a positive deadline up to an hour, or a `signal` that is not a real `AbortSignal` is `invalid_request` in identity order, with no request sent. `kept` is set only by a step with a threshold calibrated for its pinned model. Scores are not comparable across providers. `toOutcomeReranker(reranker, { kind })` wraps any rejecting `Reranker`; a ranking that is not a full permutation in descending order becomes `invalid_results`, a request's `deadlineMs` and `signal` bound the wait even for a reranker that ignores them (the deadline reaches it as its abort signal, never as `timeoutMs`, so it caps the reranker's configured timeout and never extends it), and each retry it reports through `RerankRequest.onRequest` counts as a call. A reranker that sets `reportsRequests` (System One's batch mode over the SDK's HTTP model does) is counted only from those reports, so a call that failed before sending counts nothing.

`usage` counts requests (`calls`, retries included: for a decision model with `reportsRequests`, as the SDK's HTTP model and `withBreaker` over it have, System One counts exactly the requests it reported through `DecisionRequest.onRequest`, so a `decide()` that failed before sending, such as a token minter that failed, returned nothing, or was still pending at the deadline, counts nothing and leaves a `skipped` attempt; a model that reports none counts one request per `decide()`), reported `inputTokens`, `outputTokens`, and `searchUnits`, cache hits, and `estimatedTokens`/`estimatedSearchUnits` for requests a deadline abandoned: a deadline aborts the client, not the host, which still bills the request. Only a request still in flight (or whose successful body was still being read) is estimated: when the deadline fires after a request ended with a 429 or 5xx reply and before its retry was sent, nothing is estimated and the step reports that reply's reason (`rate_limited` for a 429) rather than `timeout`. Only the latest reply counts: one followed by a 401, a dropped connection, or a score no longer describes a backoff, and in System One's document mode each document keeps its own, so another document's score does not hide it. `estimatedUsage(kind, documents, tokensPerDocument?)` is that estimate (600 tokens per System One document and 200 per cross-encoder document by default, from Codegraph's file summaries; one search unit per 100 cross-encoder documents, counted from the documents alone, so a zero token estimate still charges them). `estimatedTokensPerDocument` on the cross-encoder and on document-mode System One overrides the token estimate; anything but a finite number of at least 0 throws at construction rather than charging nothing. The SDK ships no prices: `rerankCost(outcome, prices)` prices every attempt from a caller-supplied table keyed by kind and model (`'jev-*'` patterns and `'*'` are allowed) and lists the attempts it could not price instead of counting them as free. An attempt is priced only when its entry has a rate for every reported dimension it used (`inputPerMTok`, `outputPerMTok`, `perSearchUnit`; `0` marks a dimension free, while an absent or invalid rate leaves it unpriced). An abandoned request's estimate books both tokens and search units, since the SDK cannot tell which the host bills, so it is priced by whichever of those two rates the entry has.

### System One

```ts
import {
  createDecisionModel,
  decisionPreset,
} from '@librechat/agents/decisions';
import {
  createSystemOneReranker,
  createCrossEncoderReranker,
  noneReranker,
  withFallback,
  probePin,
} from '@librechat/agents/rerank';

const jevFor = (model: string) =>
  createDecisionModel({ ...decisionPreset('clickhouse')!, model }, credential);

const jev = createSystemOneReranker({
  model: jevFor('jev-1.13.0'),
  rubric: [
    'Unrelated to the query',
    'Relevant context but does not answer the query',
    'Directly answers the query',
  ],
  pin: { model: 'jev-1.13.0' }, // add `threshold` only when calibrated for this rubric
});

const outcome = await jev.rank({ query, documents, deadlineMs: 900 });
```

By default (`mode: 'document'`) the adapter asks **one score question per candidate, one request each**, with the state `{ query, document }` (override with `state` and `instructions`; the instructions and rubric an evaluation used are part of what it measured). Requests run under a semaphore (`concurrency`, default 64, or a shared `semaphore`). Candidates left unscored by a missing answer or a failure of their own are re-issued once inside the remaining deadline. The deadline (the lesser of `deadlineMs` and `timeoutMs`, default 10 seconds, whole milliseconds up to an hour or the constructor throws: a request's deadline caps the configured timeout, never extends it) counts from the call, so building custom states and cache lookups spend it, and no request is sent after it; a ranking the cache answers in full is still served. `instructions` are snapshotted at construction, so mutating the object afterwards changes neither the question nor its cache key. `unauthorized`, `quota`, `rate_limited`, an open breaker, an unsupported question, and a pin mismatch stop the step: the rest are never sent and nothing is re-issued. For a decision model that reports its requests (the SDK's HTTP model), a 429 stops the step on the reply rather than after the transport's retries, and a document between requests (minting a token, backing off from a 5xx or from a 2xx whose body broke off, refreshing after a 401) is halted, as is any retry that starts after the stop, so nothing is retried or re-sent and a throttled host gets no extra traffic; a request already on the wire may finish. `rerank()` then rejects with the error of the failure that stopped the step first, a 429 with its `retryAfterMs`. An open breaker refuses before dispatch, so its refusals are not counted as `calls` or estimated, and a step that sent nothing is a `skipped` attempt rather than a `failed` one. The outcome is `ok` only when every candidate has a score; otherwise it is `skipped` with `timeout`, `aborted`, `partial`, or the documents' own failure, and what was scored is cached and charged. With `pin`, every response must echo the pinned model or the step is `pin_mismatch`; a pin that is not the decision model's configured model is refused without a request. `cache` (`createScoreCache()` or a host store) reuses scores keyed by query, document text, model, rubric, and, when a custom `state` builds the request, that serialized state (with a cache, the request sends a snapshot of that state, so mutating the returned object after `rank` changes neither), and stores only scores the configured model answered. A cached value outside the rubric's range is a miss. `createScoreCache({ max?, ttlMs? })` throws on a `max` that is not a positive integer or a `ttlMs` that is not a positive finite number, since either would leave it unbounded. `rerank()` on the same object keeps the rejecting contract: it resolves only when every candidate is scored and otherwise rejects with the underlying `DecisionError`.

`mode: 'batch'` keeps the earlier single request with one shared state and one question per candidate. Its scores follow input order: in Codegraph's evaluation (paraphrased developer questions, n=329) the same batch ranking reached hit@1 22.8% with candidates in embedding order and 5.8% with the same candidates reversed. Per-document Jev (jev-1.13.0) reached hit@1 23.4–23.7% over two runs against 7.6% for the embedding order. Use batch mode only where its order dependence was evaluated. Batch mode takes no pin, cache, state, instructions, concurrency, semaphore, or `estimatedTokensPerDocument` options.

`probePin({ modelFor, pin, sentinel? })` checks that a host really serves a checkpoint before a pinned step is trusted: a sentinel model no host serves must be refused with HTTP 404 (a host that answers it, as Laya does, echoes whatever it is asked, so its echo proves nothing), and a probe with the pin must echo exactly the pin. A `sentinel` that is blank or is the pin, a `signal` that is not a real `AbortSignal`, or a `timeoutMs` that is not an integer from 1 to 3,600,000, is `invalid_request`, with no request sent. Each request is bounded by `timeoutMs` (default 15 seconds) and the caller's `signal` around the model call, so a model that ignores both cannot hold the probe. It never throws; a refused credential or a dead host is reported as its own reason, not as a verdict on the pin, and `sentinelStatus` is only ever a status the sentinel request got, so an open breaker (`circuit_open`) reports none.

### Cross-encoder, BM25, and none

`createCrossEncoderReranker({ url, model, apiKey?, requiresAuth?, timeoutMs?, maxRetries? })` speaks the Cohere rerank shape: one `POST { model, query, documents }` answered by `results[{ index, relevance_score }]`, which must score every document exactly once. Cohere, Jina, Infinity, and the ClickHouse gateway speak it; TEI's `/rerank` (`{ query, texts }` answered by a bare array) does not. Responses are capped at 256 KiB, so a host that echoes each document in its results (Jina's default) fails a large shortlist as `bad_response`. It reads `meta.tokens.input_tokens`, `meta.billed_units.search_units`, or `usage` tokens, and echoes the response's `model`. It shares the decision transport, so a token minter (for example `commandToken`) is re-minted once after a 401, redirects fail rather than forward the key, and 402 is `quota`. Retries default to zero so a step deadline is spent on one request. A request's `deadlineMs` caps `timeoutMs` (default 30 seconds; whole milliseconds up to an hour, anything else throws at construction) and never extends it. Without a `url` or a required key the step is skipped without a request. The ClickHouse gateway serves `qwen3-reranker-8b` and `cohere-rerank-v3-5` at `POST {GATEWAY_URL}/rerank` (no `/v1`).

`createBm25Reranker({ idf, avgdl, k1?, b?, tokenize? })` is in-process Okapi BM25 and costs nothing. Scoring is synchronous: an already-aborted `signal` is `aborted` before any work, but neither an abort nor `deadlineMs` stops a pass once it starts. Corpus statistics are required: build them with `corpusStats(documents)` over the whole corpus the shortlist is drawn from; idf over a shortlist alone is close to degenerate. `tokenizeCode` splits camelCase, snake_case, kebab-case, dotted names, and paths, lower-cases, and drops stop words and words of two letters or fewer. It reads only ASCII letters and digits: CJK text yields no terms and accented words split into fragments, so pass your own `tokenize` (to both `corpusStats` and the reranker) for other scripts. A query left with no terms is `skipped` as `unsupported`, not an all-zero ranking, so a chain moves on. `noneReranker({ reason? })` is identity order with no request: one `skipped` attempt with that reason (`disabled` when it is missing or empty) and no usage. A malformed request is still `invalid_request`, and no documents `empty`, as for every other reranker.

### Fallback chains

```ts
const ranker = withFallback(
  [
    { reranker: jev, deadlineMs: 900 },
    {
      reranker: createCrossEncoderReranker({
        url: `${gatewayUrl}/rerank`,
        model: 'qwen3-reranker-8b',
        apiKey: gatewayKey,
      }),
      deadlineMs: 500,
    },
    { reranker: noneReranker() },
  ],
  { gate: (step) => (inBackoff(step) ? 'backoff' : null), onAttempt: record }
);
```

Steps run in order and the first that answers wins: step 0 answering is `ok`, a later step is `degraded` with the first failed step's reason, and none answering is `skipped` in identity order with that reason. A request's `deadlineMs` bounds the whole chain: each step gets the lesser of its own deadline and what is left, a step without one gets what is left, and once nothing is left the chain stops with `timeout`. A step's deadline caps the reranker's own `timeoutMs` and never extends it, so a System One step configured for 200 ms still falls back after 200 ms under a 1.5-second request deadline. Each step gets its own deadline as an abort signal and a timer that ends the wait `STEP_GRACE_MS` (50 ms) later, so a step that ignores its abort cannot starve the next one; the caller's abort ends the wait the same way, with or without a deadline, as `aborted`. Such a step is charged as if every document was sent, and a full answer it returns within the grace is accepted. A step that throws, rejects, or returns something that breaks the `RerankOutcome` contract (a full permutation in score order, every document scored when the status is not `skipped`, `kept` only with a `threshold` and then on every entry, `true` exactly where the score reaches it, well-formed attempts, a `reason` that is `null` exactly when the outcome's or attempt's status is `ok` and otherwise a non-empty string, and well-formed `provider`, `threshold`, and `pinVerified`) is a failed step. A step that ran but reported no attempt is recorded as one, so `attempts` keeps every step. `kept`, `threshold`, and `pinVerified` come from the answering step only, so falling back to a cross-encoder never carries a System One threshold. The chain's `usage` is summed from the steps' attempts, never from a step's own total, so it always matches `attempts`. `gate(reranker)` returning a reason skips a step without a request; a gate still pending at the request's deadline or abort is abandoned and the chain stops with `timeout` or `aborted`, and both are checked again after a gate before a step starts. `onAttempt(reranker, outcome)` sees a copy of every step that ran, so a hook that sorts or trims what it is given changes nothing the chain returns.

Related: [agents#66](https://github.com/LibreChat-AI/agents/pull/66) adds a configurable `custom` web-search reranker type with its own URL and model. The cross-encoder here is the index-based outcome layer, not a web-search reranker type, and the web-search Jina and Cohere models stay as they are so the two changes do not overlap.

## Scope and evaluation

This SDK supplies the seams and adapters. It does not enable reranking in Codegraph `/find`, add LibreChat `systemone` configuration, install Laya, pick a model, ship prices, or change search defaults.

Before enabling a System One ranker, evaluate the same queries and candidates against embedding order, BM25, pinned Jev, local Laya, gateway Qwen3/Cohere, and a cross-encoder shortlist followed by Jev. For web search, compare against the existing Jina and Cohere paths. Record relevance, calibration where applicable, latency, cost, failure rate, and candidate limits. No quality, latency, or cost advantage is claimed by the contract tests.

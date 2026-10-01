# Reranking

`Reranker.rerank({ query, documents, topK?, signal?, timeoutMs? })` returns ranked input indices and provider-specific scores. Indices preserve candidate identity even when two documents have identical text. A successful ranking is ordered by descending score. Scores are not portable confidence thresholds. Missing usage is `null`.

## Host-owned web-search ranking

`createSearchTool({ reranker })` now honors an injected `SearchReranker` before constructing its configured built-in reranker. Hosts can supply a plain `{ provider, rerank(query, documents, topK?, metrics?) }` object. Extending `BaseReranker` is not required. Existing Jina, Cohere, rag_api, Infinity, and `none` selection stays unchanged when there is no injection.

For an index-based `Reranker`, `createWebSearchReranker(reranker, { timeoutMs?, logger? })` maps indices to highlights, validates ordering/ranges/duplicates, and records exactly one observation per attempt. Invalid or failed rankings fall back to input order with neutral scores. Error metrics never copy provider exception text. Empty inputs and zero result limits require no provider work.

## Experimental System One adapter

```ts
import {
  createDecisionModel,
  decisionPreset,
  createSystemOneReranker,
  createWebSearchReranker,
  createSearchTool,
} from '@librechat/agents';

const decisionModel = createDecisionModel(
  {
    ...decisionPreset('laya'),
    baseURL: 'http://localhost:8000/v1/systemone',
  },
  undefined,
  { providerId: 'laya' }
);

const ranker = createSystemOneReranker({
  model: decisionModel,
  rubric: [
    'Unrelated to the query',
    'Relevant context but does not answer the query',
    'Directly answers the query',
  ],
  maxDocuments: 128,
  timeoutMs: 10_000,
});

const result = await ranker.rerank({
  query: 'How does request cancellation work?',
  documents: ['Candidate A', 'Candidate B'],
  topK: 1,
});

const webSearch = createSearchTool({
  // Supply the host's existing search and scraper credentials here.
  reranker: createWebSearchReranker(ranker),
  chunkSize: 1_000,
  chunkOverlap: 100,
});
```

The example rubric is illustrative, not an evaluated default. Operators supply and evaluate their own ordered relevance levels. Jev and self-hosted Laya use the existing HTTP decision model: endpoint, checkpoint, and auth are host-owned. Hosted Jev evaluations should pin a checkpoint instead of `jev-latest`. Laya need not have a bearer key or explicit checkpoint; no gateway availability is assumed.

The adapter sends one shared query/candidate state with one expected-value score question per candidate in **one decision request**, not an N+1 request loop. Ties keep input order. Every candidate must have a valid score. Partial answers fail rather than assigning missing candidates zero. Chat decision models reject score questions before invocation. One deadline bounds preparation, provider invocation, and ranking; requests keep independent signals. Candidates and rubric levels are snapshotted. The default candidate limit is 128 and oversized batches fail explicitly, never silently truncate. Hosts must align their chunking and batch limits with their evaluated budget.

## Scope and evaluation

This SDK change supplies the seam and an experimental adapter. It does not enable reranking in Codegraph `/find`, add LibreChat `systemone` configuration, install Laya, or change search defaults.

Before enabling a System One ranker, evaluate the same queries and candidates against embedding order, BM25, pinned Jev, local Laya, gateway Qwen3/Cohere, and a cross-encoder shortlist followed by Jev. For web search, compare against the existing Jina and Cohere paths. Record relevance, calibration where applicable, latency, cost, failure rate, and candidate limits. No quality, latency, or cost advantage is claimed by the contract tests.

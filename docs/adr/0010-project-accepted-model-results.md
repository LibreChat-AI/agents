# ADR 0010: Project Accepted Model Results, Not Provider Fragments

## Status

Proposed in Agents SDK PR #560. Implemented on that branch, not released or adopted by LibreChat.

## Context

The first opt-in OpenAI projector copied LibreChat's UI run-step reconstruction.
It attempted to infer identity, delta versus snapshot semantics, successful model
attempts and completion. Review found two classes of P1 that local stream tests
could not rule out: complete snapshots lost to buffered partial data, and
invoke-only final results lost because their run-step metadata lacked an attempt
stamp. An API formatter cannot recover information that its input events omit.

LangChain.js's newer content-block event contract distinguishes deltas from
finished content. Its legacy AIMessageChunk.concat is not a generic normalizer of
arbitrary cumulative snapshots. This change adopts the explicit acceptance
boundary, not a wholesale migration to a different provider API.

## Decision

The graph emits a registry-only ON_MODEL_RESPONSE event from its common accepted
result path. Failed primary/fallback attempts and overflow detours emit none.
Invoke-only and streaming results converge here after usage accounting. The
payload contains finalized native ToolCalls and invalid_tool_calls; a graph-made
acceptance ID avoids treating provider IDs or UI indexes as invocation identity.

The event is awaited. Exceptions propagate outside the provider fallback block,
so projection failure cannot rerun a provider or tools. Calls are detached before
host delivery so observers cannot mutate tool arguments awaiting execution.
Generic provider/tool custom events cannot impersonate this graph-only event.
Child graphs retain existing narrow handler forwarding; the parent projector is
not inherited as a child output sink.

The OpenAI projector accepts only this event, serializes valid calls once, and
assigns outward indexes. No fragments, attempts, merge heuristics, or missing-ID
fallbacks remain. Identity lookup is set-based; synthetic ID generation uses a
monotonic counter. Text-only responses retain no delivery-ID state. Retention is
O(accepted calls + output bytes), with limits on both calls and encoded bytes.
The graph event clone still costs O(finalized result size). Argument encoding
accepts only JSON data, without executing serialization hooks or getters. It rejects
exotic objects rather than silently converting them, supports cross-realm plain
objects, and bounds both nesting and encoded bytes during traversal. The per-run
byte bound includes the expansion of repeated references; cycles fail.

Streaming projection and the public OpenAI finalizer share one tracker. Projected
tool chunks update its last-chunk state and initialize the assistant role once.
Later assistant text can still restore a stop finish reason. Map-only collection
remains available for non-streaming/custom finalization. Hosts must not also attach
legacy raw tool handlers to this path.

Output waits for host-confirmed natural run completion. The host must abort on
interrupt, halt, disconnect or exception, not equate promise resolution with
success. Emission checks cancellation around every synchronous callback and
cannot retry after partial publication. The helper is per-response and is not a
checkpoint delivery log. Existing eager tools can have side effects before
acceptance; no rollback or exactly-once execution is promised.

## Alternatives and trade-offs

- More string/snapshot/attempt reconciliation adds guesses, not evidence. Rejected.
- Replacing every provider with LangChain's native event API is a separate migration
  and still requires a graph-level accepted-attempt boundary. Deferred.
- Finalized accepted results fit complete-before-publish behavior, at the cost of
  not providing incremental tool arguments. Existing default handlers remain intact.

Only this PR's unreleased opt-in API changes. No published entry point is removed,
no storage schema changes, and no dependency version bump occurs here. LibreChat
must integrate a released SDK and compare its HTTP/JSON contracts separately.

## Verification obligations

Use real Run.processStream, attemptInvoke, graph tool execution and LangChain
callbacks, replacing only provider transport. Cover streaming/invoke-only, partial
primary then tools/text fallback, overflow retry, cancellation, invalid final
results, observer errors and mutations, callback spoofing, concurrent runs, usage
accounting and limits. Preserve existing default OpenAI/Responses and graph lifecycle
suites. Check TypeScript, lint, build, CJS/ESM exports and the public declarations.

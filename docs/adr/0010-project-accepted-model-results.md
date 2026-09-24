# ADR 0010: Project Accepted Model Results, Not Provider Fragments

## Status

Proposed in SDK PR #560. Not released or adopted by LibreChat.

## Context

Run-step events mix partial data, snapshots and failed attempts. They cannot
reliably identify the accepted result across streaming, invoke-only and fallback
paths. More formatter heuristics cannot recover missing execution information.

## Decision

The graph emits an awaited, registry-only `ON_MODEL_RESPONSE` after acceptance,
fallback/overflow recovery and usage accounting. Model outputs validate and
detach parsed calls and raw fragments before stream accounting or dispatch,
without imposing projection budgets on ordinary runs. Composed observers receive
isolated snapshots. Provider/tool custom callbacks cannot impersonate acceptance. Handler errors
propagate outside provider retry logic.

The opt-in OpenAI projector buffers finalized calls, reserves provider IDs, and
formats output after host-confirmed natural completion. A registry-only ToolNode
claim retires graph-owned batches by agent and message identity, including
terminal tool runs. Only unclaimed calls appear as client `tool_calls`. Fresh
output state and bounded pending call count, encoded bytes and depth prevent
unbounded retention. See the
[README](../../README.md#accepted-tool-call-projection-opt-in) for registration,
limits and failure handling.

## Trade-offs and verification

This removes fragment/attempt reconstruction but delays tool-call output. It does
not replace provider normalization, sandbox callbacks, roll back eager tools or
provide durable delivery. Existing default handlers remain unchanged; LibreChat
integration and release are separate gates.

Verify through real `Run.processStream` and the public finalizer: streaming/invoke,
fallback/overflow, final-answer ordering, observer isolation, cancellation,
malformed arguments, output limits, subagents, usage and tracing.

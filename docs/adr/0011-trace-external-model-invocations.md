# ADR 0011: Trace Host Model Invocations Through an Owned SDK Lifecycle

## Status

Accepted

## Context

Hosts execute model calls outside a graph `Run`, such as conversation titles and
bounded provider transports. Those calls need the same tenant destination,
identity, sampling and callback policy as SDK model calls. Exporting the raw
Langfuse handler factory, attribute wrapper and disposal helpers would make every
host responsible for the SDK's internal handler lifecycle.

## Decision

Expose `traceModelInvocation(params, work, project?)`. It initializes the resolved
tracing destination before constructing a handler, scopes callbacks to the
invocation and disposes the handler afterward. LangChain callers receive callbacks
directly. External clients provide a result projection so one generation
observation is created without implementing Langfuse callbacks themselves.

The host decides which result content may be exported. The SDK never records an
external client's request body or raw error: it records a generic failure and
rethrows the original error. A failed projection completes the observation with
an omitted-output marker.

## Consequences

Initialization, export and projection failures cannot repeat inference or replace
a successful paid result. The handler factory, attribute wrapper, disposal
function and tracing initializer remain internal to the package, so their
signatures can change without a host-facing break.

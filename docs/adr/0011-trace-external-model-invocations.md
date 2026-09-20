# Trace host model invocations through an owned SDK lifecycle

Hosts also execute model calls outside a graph Run, including titles and bounded
provider transports. Those calls need the same tenant destination, identity,
sampling and callback policy as SDK model calls. Exporting the raw Langfuse handler
factory and disposal helpers made hosts responsible for the SDK's internal lifecycle.

Expose `traceModelInvocation(params, work, project?)` instead. It initializes the
resolved tracing destination before constructing the handler, scopes callbacks to
the invocation, and disposes the handler afterward. LangChain callers receive
callbacks directly. External clients provide a result projection to create one
generation observation without implementing Langfuse callbacks themselves.

The host decides which result content may be exported. The SDK never records an
external client's request body or raw error. It records a generic failure and
rethrows the original error to the caller. Initialization, export and projection
failures cannot repeat inference or replace a successful paid result. A failed
projection completes the observation with an omitted-output marker.

The public package does not export the internal handler factory, attribute wrapper
or disposal function. This API requires an actual SDK release before a host can
remove its temporary development bridge.

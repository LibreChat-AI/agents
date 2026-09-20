# Persist native media before model output

Status: accepted for the native-media port proposal.

Native image bytes and thought signatures arrive in provider response parts. The
Google adapter calls the injected host port before returning each persisted part.
Only text and durable file references enter graph state, in-run replay, Langfuse,
SSE, or subagent results. Moving persistence after model output would expose raw
bytes or a file reference that another consumer can observe before it exists.

The port owns authorization, storage and continuation restoration; the adapter
owns provider parsing and ordered emission. Waiting for one part at a time
provides bounded backpressure without a queue of unpersisted image bytes. Hosts
should skip persistence for unsigned text and bound every storage operation.
Cancellation can leave a persisted image, so hosts reconcile incomplete writes;
an aborted stream does not prove the provider stopped generating or billing.

Response policing is tied to the invocation's explicit IMAGE admission. An
attached port can restore history for a text model without changing ordinary
empty, blocked, or malformed-function-call response behavior or adding a modality
selection. Usage-bearing failures use a provider-neutral error contract consumed
by shared tracing; provider details remain optional native-media metadata.

Vertex native output remains a separate extension. It requires raw-part
interception at the Vertex connection boundary and a host binding scoped to the
service account, project and location. The current port supports the Gemini
Developer API; Studio Vertex adapters are independent of this SDK port.

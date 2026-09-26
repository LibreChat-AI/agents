# ADR 0010: Persist Native Media Before Model Output

## Status

Accepted

## Context

Gemini image models return inline image bytes and thought signatures as ordinary
response parts. Anything the Google adapter yields enters graph state, in-run
replay, Langfuse observations, SSE output and subagent results. Raw bytes must not
reach those consumers, and a file reference must not be visible before the file
exists. The host, not the SDK, owns authorization, storage, retention, accounting
and recovery for generated media.

## Decision

`CustomChatGoogleGenerativeAI` accepts an injected `NativeMediaPort`. The adapter
calls `port.part` for each text or image part and waits for a durable reference
before yielding it, so only text and file references enter the stream. Waiting on
one part at a time gives bounded backpressure without a queue of unpersisted
bytes. The port owns authorization (`start`), persistence (`part`), completion and
failure recording (`complete`, `fail`) and continuation restoration (`restore`,
`restoreBatch`); the adapter owns provider parsing and ordered emission.

Response policing is tied to the invocation's explicit `IMAGE` admission. A port
attached to a text model can restore history without changing empty, blocked or
malformed-function-call handling and without adding a modality selection.
Failures that already consumed provider tokens raise a provider-neutral
`UsageBearingError` so shared tracing and the host can account for them.

## Consequences

Hosts must bound every storage operation, skip persistence for unsigned text, and
reconcile incomplete writes after cancellation: an aborted local stream does not
prove the provider stopped generating or billing. Continuation references must
survive later invocations and process restarts when conversations can be resumed.

Vertex native output is a separate extension. It needs raw-part interception at
the Vertex connection boundary and a host binding scoped to the service account,
project and location. This port supports the Gemini Developer API.

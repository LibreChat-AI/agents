# Host-owned models in SDK graphs

`HostChatModel` is a native LangChain chat model backed by a versioned
`HostModelService`. A custom SDK graph can use the facade in its normal model
node while a host executes its configured provider model. Builder model selection,
provider credentials, model parameters, input/output protections and host callbacks
stay with that host model. The worker selects only approved function tools and
receives native assistant messages/chunks, including reasoning, tool fragments,
usage and provider metadata.

This is the SDK contract and consumer facade. It does not install a LibreChat
transport or enable protected saved-agent execution by itself. Keep unsupported
policy guards until the host integration passes its own acceptance tests.

## Worker integration

```typescript
import { HostChatModel } from '@librechat/agents';

const model = new HostChatModel({
  identity: authenticatedExecutionIdentity,
  descriptor: negotiatedModelDescriptor,
  service: authenticatedModelTransport,
});

// Use in a custom LangGraph node, or register a provider backed by this class.
// Pass the same approved definitions used to construct the descriptor.
const withTools = model.bindTools(approvedFunctionTools, {
  tool_choice: 'auto',
});
const response = await withTools.invoke(messages, { signal });
// Or stream without buffering a whole response:
for await (const chunk of await withTools.stream(messages, { signal })) {
  consumeNativeChunk(chunk);
}
```

A descriptor contains `version: 1`, an opaque `reference`, public `provider` and
`model` labels, `streaming`, and canonical OpenAI-format function `tools`. The
reference identifies a pre-bound host model, not a URL or credential. The facade
snapshots identity and definitions. `streaming: false` routes `.stream()` through
invocation for models that cannot stream.

`bindTools` accepts structurally identical approved schemas, selects a subset,
and normalizes `auto`, `none`, `required`/`any`, and forced tool names. A forced
name must belong to that subset. Unknown tools, changed schemas and duplicate
names fail before contacting the service. An invocation's `tool_choice` can
override a bound default. Provider settings, stop sequences, arbitrary
`configurable` data, metadata and callback objects are not sent to the host.
Worker callbacks remain local; host callbacks remain attached to the actual model.

The native SDK provider registry can construct the facade through its normal
model initialization and tool-binding path. Register a stable provider once,
resolving only authenticated, run-scoped model bindings in its constructor; do
not register a new global provider name for every run. Choose provider-family
traits to match the host model and test provider-specific message projection.
The SDK tests also run the normal graph/tool loop through a registered provider.

## Native host binding

`createHostModelService` supplies the host-side binding for an already configured
native model. Reuse `HostModelAdmission` across all bindings sharing a process-local
quota; do not create a new scope per request or reconnect.

```typescript
import { createHostModelService, HostModelAdmission } from '@librechat/agents';

const admission = new HostModelAdmission({
  maxConcurrentCalls: 32,
  maxRequestBytes: 8 * 1024 * 1024,
});

// Retain for this authenticated execution. The model is created by the host
// using builder settings, credentials and the deployment's policy configuration.
const binding = createHostModelService({
  identity: authenticatedExecutionIdentity,
  descriptor: negotiatedModelDescriptor,
  model: configuredNativeModel,
  admission,
  limits: {
    timeoutMs: 120_000,
    maxRequestBytes: 1024 * 1024,
    maxResponseBytes: 1024 * 1024,
  },
  config: { signal: hostRunSignal, callbacks: hostCallbacks },
});
```

The returned service compares every identity field and model reference, validates
selected tools again, and binds only its private snapshot of approved definitions.
The worker cannot replace schemas or host settings. Each provider operation runs
inside the explicit host configuration, isolated from a caller's ambient LangChain
callbacks, metadata and configurable overrides. Tool-choice encoding uses the
SDK provider registry for OpenAI, Anthropic, Bedrock and Google families. Required
choice uses the native `any` alias; Bedrock `none` removes the tool definitions
because Converse has no `none` choice. Register custom providers with accurate
family traits. For an unregistered generic model, forced reserved names (`auto`,
`none`, `any`, `required`) fail closed instead of being interpreted as a different
choice. An empty selection explicitly
binds an empty tool list when the descriptor contains tools. Native messages are
snapshotted through the SDK's `toStoredMessage` representation before asynchronous
provider dispatch. The same helper is exported for transport/checkpoint adapters:
it reads current native message fields without LangChain's JSON-oriented escaping,
so binary data, dates, maps, sets, tool artifacts and caller tag fields reach the
adapter's codec intact. It returns field references, not a deep copy or a wire
encoding; the adapter must apply its own bounded snapshot/encoding immediately.
Custom messages declaring `lc_secrets` fail closed and require an explicit codec;
this helper does not silently bypass their secret-redaction contract.
A chunk-aware codec must preserve chunk kind separately, because the standard
LangChain stored-message decoder constructs full messages. Host-local V8 serialization preserves binary data, dates,
BigInts, maps and sets instead of coercing them through JSON. Functions and other
uncloneable values fail closed. The returned descriptor is a separate copy.

The host's execution signal and the invocation signal are combined. Each operation
has a required deadline, a per-request message-byte limit, and a cumulative response
byte limit (including rich content, metadata and tool fragments). Shared admission
bounds concurrent calls and total charged request-message bytes. Limits count
the host-local binary serialization of native messages, not a transport wire format: the transport still
needs frame and queue bounds before decoding. Provider-internal caches/buffers and
cluster-wide quotas are outside this process-local admission scope.

Cancellation/deadline expiry stops the caller's wait promptly. Admission remains
charged while an invocation, outstanding stream read, or iterator cleanup has not
settled, including awaited host terminal callbacks. `snapshot()` exposes total and retained ownership without content. There
is no reset that can free live work. A deadline also closes a stream whose consumer
is paused between chunks. Underlying models must faithfully expose settlement of
their provider work; this binding cannot discover work a custom model hides after
its own promise/iterator has settled.

The binding uses native `invoke` and `_streamIterator` with configured callbacks.
The latter is the same model/callback path underlying LangChain streaming, without
the `ReadableStream` adapter whose cancellation can return before asynchronous
provider cleanup finishes. Tests cover both normal tool binding and a provider
that ignores cancellation and blocks its cleanup. The binding does not call the
provider's raw `_generate` or `_streamResponseChunks` methods.

An early iterator return can bypass LangChain's terminal callback. The binding
closes that native run once, after provider cleanup, using the SDK's shared callback
composition helper. It preserves the observed run ID, parent and tags, records
partial output as `HostModelTerminated` control flow, and preserves reported usage.
Usage is accumulated separately from released content, including a usage-bearing
chunk rejected by the response-byte limit; it appears in native message metadata
and `llmOutput.tokenUsage` without releasing that rejected content.
Missing usage is marked `usage_incomplete`; no token counts are invented. Normal
native success/error callbacks are not duplicated. Provider billing reconciliation
for cancelled requests remains a host integration responsibility.

## Host service boundary

The service exposes `invoke(request, signal)` and a pull-based
`stream(request, signal)`. Each request contains only protocol version, execution
identity, model reference, native messages, selected tool names and normalized
tool choice. It does not carry a provider endpoint or worker-supplied model
configuration.

The native binding supplies identity/reference validation, tool selection, message snapshots,
admission and deadlines. A production transport/host integration must additionally:

1. Authenticate the transport, resolve its pre-bound execution, and compare every
   identity field and model reference. Identity supplied in a request is not
   authentication. Validate shape, message kinds, content sizes and selected tools
   again on the host; worker checks are not an authorization boundary.
2. Bind the selected tools from its own approved definitions to the already
   configured model. Use that model's public `invoke`/`stream` APIs so its native
   callbacks and protections execute. Return only output the host policy permits
   releasing. A callback that only records an error is not an output-release gate.
3. Keep credentials, builder settings, policy configuration and trace exporters
   out of the descriptor. Public labels should describe the actual model. Attribute
   usage on the host once, using the bound execution and model attempt identity.
4. Enforce shared admission, request/response byte limits, deadlines and bounded
   transport queues. Preserve backpressure. Streaming `next()` advances only when
   requested; a transport must not hide an unbounded producer behind that iterator.
5. Correlate attempts with the transport's operation identity. Preserve native
   message types, rich content, usage and provider metadata with an explicit codec.
   Native message objects here are an in-process contract, not a JSON wire schema.
6. Forward cancellation and iterator cleanup to the actual provider. Retain
   ownership/admission until underlying work drains, even when the caller has
   stopped waiting. A timeout or disconnected consumer does not prove that a
   provider request stopped or that billing did not occur.
7. Negotiate protocol support before graph execution. Never fall back to a local
   unprotected model when a host service is unavailable.

The facade adds no automatic retries. A failed invocation may already have
incurred provider work, so retry/replay semantics belong to the transport and
host. It rejects promptly on cancellation and observes late settlement without
releasing late output. Stream cleanup requests iterator return without blocking
on a provider that ignores cancellation; the host still owns that outstanding
work.

`HostModelServiceError` exposes only an allowlisted code: `incompatible`,
`forbidden`, `blocked`, `overflow`, `timeout`, `cancelled`, or `unavailable`.
Unexpected service failures become `unavailable`; raw messages/stacks are not
forwarded to worker callbacks. Transport adapters must explicitly encode and
validate failures. Local SDK cancellation reasons for stream limits and delegated
work retain their control-flow semantics.

## Coverage and remaining integration

Tests exercise a real configured host model, its callbacks, native SDK graph
and tool execution, nonstreaming models, tool-choice override, rich content/tool
fragments/usage, backpressure, cancellation while blocked, consumer termination,
late rejection, safe failures, and immutable identity/schema snapshots. Native binding tests
add authorization checks, message snapshots, shared quotas, cumulative response limits,
host-owned cancellation, idle-stream deadlines and retained ownership through cleanup.

The facade and native binding do **not** establish complete remote-chat parity. Remaining work:

- LibreChat authenticated transport and recovery, capability negotiation, and
  integration of the native binding with builder policy/protections and billing.
- End-to-end protected deployments with provider-specific callbacks and message
  projections, cancellation/approvals/reconnection, and measured UI latency.
- Native provider tools that are not function schemas, host-approved dynamic tool
  discovery/schema changes, child model subjects and model-fallback authorization.
- Host-authoritative context management and trace correlation/redaction across
  processes. The facade preserves native usage and labels; that alone does not
  establish accounting or tracing parity.
- Files/artifacts, execution steering/composition, and product integration.

Arbitrary trusted worker code can still call another model directly. This facade
is an integration boundary, not a sandbox or proof that a custom graph uses the
host model for every generation.

## Invocation correlation through output protection

Host model services may accept a third argument on `invoke` and `stream`:
`context?: Readonly<ModelInvocationContext>`, containing only `{ version: 1, runId }`.
The ID matches the native model callback run ID, including when provider-text
protection suppresses the underlying callback manager. `HostChatModel.invocationContextVersion`
advertises support as `1`; older two-argument service implementations remain compatible.

The provider-text boundary carries context in an internal weak map keyed by the
specific invocation options object. It does not add provider parameters, callbacks,
message content, or ambient async context. Concurrent and nested protected calls
retain their own IDs, and context is not serialized automatically. Direct low-level
calls without a native callback manager or a protected invocation have no context;
transports requiring correlation must reject those calls rather than invent identity.

This ID is correlation, not authorization. Transport adapters must still authenticate
the execution identity, correlate it with their own root graph callbacks, validate
it before accepting lifecycle events, and negotiate support before running a graph.
The raw callback manager remains suppressed: only approved provider output reaches
public callbacks, events, and persistence. This contract does not itself implement
remote admission, tracing, or billing.

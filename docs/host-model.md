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

## Host service boundary

The service exposes `invoke(request, signal)` and a pull-based
`stream(request, signal)`. Each request contains only protocol version, execution
identity, model reference, native messages, selected tool names and normalized
tool choice. It does not carry a provider endpoint or worker-supplied model
configuration.

A production host adapter must:

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
late rejection, safe failures, and immutable identity/schema snapshots.

This slice does **not** establish complete remote-chat parity. Remaining work:

- LibreChat authenticated model service, bounded streaming transport and recovery,
  capability negotiation, builder binding, host protection and billing enforcement.
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

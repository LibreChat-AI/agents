# Host tool bindings

For authorized attachment reads and private artifact staging/publication, see
[Host files and artifacts](./files.md).

`createHostToolBindings` connects the SDK's existing tool/approval lifecycle to a host service.
A descriptor combines the model-visible `LCTool` schema with an opaque authorized host reference.
The same service interface supports direct calls and remote RPC; the SDK never selects URLs,
credentials or executable modules from model arguments.

```typescript
const bindings = createHostToolBindings({
  identity: authenticatedIdentity,
  tools: [{ definition: inspectSchema, reference: 'schema-advisor:inspect' }],
  service: {
    version: 1,
    invoke: (request, signal) => authorizedHost.execute(request, signal),
  },
});
const run = await Run.create({
  ...config,
  customHandlers: { ...config.customHandlers, ...bindings.customHandlers },
  graphConfig: {
    ...config.graphConfig,
    toolDefinitions: bindings.toolDefinitions,
  },
});
```

Supply the SDK's existing hooks, HITL configuration and checkpointer normally. Approve, edit and
reject decisions pass through ToolNode before the host is invoked. Keep one execution owner for
`ON_TOOL_EXECUTE`; do not compose this handler with another executor for the same batch. Other
custom handlers remain ordinary SDK handlers. Custom graphs can install the same bindings on
their SDK agent nodes; direct graph tools keep their existing SDK execution path.

The host establishes identity before creating bindings. A remote service must authenticate its
caller and compare the request identity to that authenticated authority, authorize each reference,
and validate arguments against its authoritative schema. The identity field is context, not proof
of authentication. No arbitrary invocation configuration, user credentials or closures are sent.
Child execution lineage and the SDK caller-capability projection are preserved.

Execution failures are surfaced through the normal SDK batch rejection path with a sanitized
message. Results must match every requested call exactly once; they are delivered in request
order. Cancellation is forwarded, and cancelled results are not accepted. The service must honor
the signal and enforce deadlines. This module does not retry a failed execution. A transport must
reconcile uncertain side effects before retrying; a call ID alone does not prove exactly-once
external execution. Existing SDK result types and native events remain authoritative.

This is the first shared host-service capability. It does not yet wire LibreChat's remote worker
to host tools, add remote model-policy enforcement, or claim complete feature parity. The HTTP
contract tests deliberately use loopback without production authentication; production bridge
adapters must supply that authority and the existing bounded/recoverable transport.

Services may accept a third `HostToolEvents` argument and publish individual results through
`events.onResult` while the batch is running. It is supplied only when ToolNode permits early completion (approval/result
policy may disable it). The final returned array remains authoritative and must contain all
results. Duplicate early receipts are suppressed; invalid identities fail the batch, and late or
cancelled callbacks are ignored. Callback messages must be delivered before settling the batch
RPC, so a final response cannot overtake them on a multiplexed transport.

`request.identity.agentId` is the authenticated root binding. `request.executingAgentId` is the
SDK execution subject from the batch, and `executionContext` carries child lineage. Hosts must
authorize the execution subject against that root binding; the two IDs must not be conflated.

`events.onSessionUpdate({toolCallId, context})` transports host-refreshed code session state.
Local services that update their request also retain that state; the harness reconciles only the
SDK session fields, including when execution fails. Remote adapters must deliver session updates
before settling either a successful or failed invocation. This preserves lazy input provisioning
and sandbox refresh semantics without copying request credentials or arbitrary configuration.

## Host provider-text protection

`createHostProviderTextService` and `createHostProviderTextProtection` connect the existing SDK
provider-text release boundary to a host-owned policy. The host keeps classification, inspection,
credentials, deadlines and aggregate raw-text budgets. The worker receives a versioned limits
descriptor and an authenticated service implementation. The SDK's normal protected model path
buffers each complete attempt and releases only the canonical text returned by the host; it does
not issue an RPC per token.

```typescript
// Authenticated host: retain this service for the bound execution.
const { descriptor, service } = createHostProviderTextService({
  identity: authenticatedIdentity,
  policy: hostProviderTextProtection,
});

// Worker: the transport forwards cancellation and maps typed policy errors.
const providerTextProtection = createHostProviderTextProtection({
  identity: authenticatedIdentity,
  descriptor,
  service: authenticatedPolicyTransport,
});
const run = await Run.create({ ...config, providerTextProtection });
```

Transport endpoints must validate the request shape, authenticate the connection, select its
pre-bound service, enforce frame/queue limits, propagate invocation cancellation, and negotiate
capability support before graph execution. Identity data is not authentication. Service calls
reject a mismatch in any execution identity field before calling the host policy. Use the versioned error
helpers described below to reconstruct typed policy failures on the caller and sanitize all other
failures. This SDK layer does not choose endpoints, authenticate
connections, retry calls, or journal raw candidates. A transport must explicitly account for
candidate retention in its own memory, persistence and tracing policy.

Host limits are enforced even if a worker enlarges its local descriptor or calls the service
directly. Bindings sharing one policy object share its host byte budget, including outstanding
inspection that ignores cancellation. Timeout/cancellation can reject a request before an inspector
settles; its budget remains charged until the inspector actually drains. Inspection statistics
remain host-owned; the worker adapter records no additional replacements/categories.

This protects the SDK provider-text output boundary only. Builder model selection, input/model
callbacks, provider credentials, accounting, tracing/export policy and non-SDK model calls require
their own adapters. A trusted custom graph must install the supplied policy on every applicable SDK
Run; this service does not sandbox or attest arbitrary worker code. LibreChat's remote runtime does
not automatically enable this service until its transport adapter and required policy negotiation
are implemented. Existing unsupported-policy rejection must remain in place for those deployments.

Absent `tenantId` and `conversationId` fields must be omitted (or `undefined` for direct calls);
`null` is invalid and is never normalized to another identity. Tool and provider-text bindings use
the same exported `snapshotExecutionIdentity` helper for identity validation and immutable snapshots.

Hosts can now supply a reusable `ProviderTextAdmission` scope and a shorter
`inspectionTimeoutMs` to `createHostProviderTextService`. Admission caps concurrent inspections
and charged bytes across all bindings sharing that scope, including cancelled work that has not
settled. `snapshot()` reports total and retained ownership without exposing content. The original
policy-wide budget remains shared; neither a timeout nor a disconnect releases retained capacity.
Reuse the scope across reconnects and binding reconstruction. Omitted options preserve the previous
behavior, so shared-host transports must configure both explicitly before rollout.

Use `serializeProviderTextProtectionError` and `deserializeProviderTextProtectionError` for the
versioned failure envelope. These preserve allowlisted codes across separate CJS/ESM copies without
serializing messages or stacks. The v1 target remains model-produced message text; other targets and
child execution subjects need explicit future contracts.

See [host provider-text protection](../../docs/host-provider-text.md) for setup, the identity and
wire contract, process-local limits, retention monitoring, and recovery without freeing live work.

## Host-owned model execution

`HostChatModel` consumes the versioned `HostModelService` through native LangChain
invoke/stream and tool-binding interfaces. A custom graph can keep the actual
provider model, builder settings, credentials and model callbacks on its host
while receiving native messages and chunks. The facade restricts worker tool
selection to approved schemas and preserves usage, rich content and cancellation.

See [host model integration](../../docs/host-model.md) for the SDK contract, required
host authorization/lifecycle checks, supported function-tool scope and remaining
LibreChat transport work. This contract alone does not enable protected remote
saved agents or establish full chat parity.

`createHostModelService` binds an already configured native host model to one
execution, validates identity/reference/tool selection, snapshots messages and
enforces required deadlines and response limits. Reuse `HostModelAdmission` across
bindings to bound concurrent calls and aggregate request bytes. Cancellation stops
the caller's wait; capacity remains charged until native work and cleanup drain.
Authentication, transport recovery and application policy/billing remain host
integration responsibilities.

# Host tool bindings

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

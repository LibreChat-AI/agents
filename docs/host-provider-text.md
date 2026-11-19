# Host provider-text protection

`createHostProviderTextService` binds an authenticated execution identity to a
host-owned output policy. `createHostProviderTextProtection` installs the SDK's
normal release boundary in the worker: it buffers a complete model attempt,
makes one inspection request, and releases only canonical approved text through
the existing event handlers. Classification, inspection, and the shared policy
memory budget remain authoritative on the host.

## Admission and deadlines

```typescript
import {
  ProviderTextAdmission,
  createHostProviderTextService,
  createHostProviderTextProtection,
} from '@librechat/agents';

// Owned by the host tenant/runtime registry, not created per request or reconnect.
const admission = new ProviderTextAdmission({
  maxConcurrentInspections: 4,
  maxBufferedBytes: 4 * 1024 * 1024,
});

const bound = createHostProviderTextService({
  identity, // authenticated runtimeId, agentId, runId, userId, optional tenant/thread
  policy, // trusted host ProviderTextProtection; keep the same object across bindings
  admission,
  inspectionTimeoutMs: 5000,
});

const protection = createHostProviderTextProtection({
  identity,
  descriptor: bound.descriptor,
  service: bound.service, // replace with the authenticated transport adapter remotely
});
// Supply protection as Run.create({ providerTextProtection: protection, ... }).
```

An admission object defines a host-selected quota scope. Use one per binding, or
share one across all of a tenant's bindings/runs so creating additional run IDs
does not bypass the quota. Reuse it across reconnects. Limits are snapshotted at
construction. Inspections fail immediately with `overflow` rather than queuing
and retaining more raw requests. Concurrency and byte limits include cancelled
or timed-out inspectors that have not settled. Byte accounting includes attempt
overhead, raw text, and canonical output; it is not merely payload length.

The original policy object's `maxBufferedBytes` still caps all its attempts,
including scopes with separate admission objects and local graph attempts. An
admission scope adds a second limit; it does not reserve capacity or guarantee
fair scheduling. Set scope caps below the shared policy cap to leave headroom
for other tenants. Limits are process-local: hosts with multiple replicas must
route a scope to its owner or provide distributed admission before dispatch.
Transport frame limits and bounded queues remain necessary before decoding
requests into this service.

`inspectionTimeoutMs` starts when the host accepts the inspection. Its effective
value is the smaller of that option and the policy's `timeoutMs`. The descriptor
continues to advertise the model-attempt timeout, which covers model generation
and inspection on the worker. The host deadline works even if worker cancellation
never arrives. It aborts the inspector signal and fails the caller; JavaScript
cannot forcibly stop a callback that ignores cancellation or blocks its event
loop. Isolation in a terminable process is required for such inspectors.

Both options are additive. Omitting admission preserves the existing shared
policy budget; omitting the inspection deadline preserves the existing timeout.
Shared-host deployments must explicitly configure both and retain admission
ownership across binding reconstruction before enabling remote inspection.

## Retention and recovery

`admission.snapshot()` returns immutable, content-free counters:

- `inspections`: all currently owned inspections, including retained ones.
- `bufferedBytes`: all charged bytes in this scope.
- `retainedInspections` and `retainedBytes`: the subset whose callers finished
  while underlying work still owns its input.

Poll these for metrics and alert when retention persists. There is intentionally
no force-release/reset API: expiry does not prove that an inspector released raw
data. Capacity returns automatically when the underlying promise resolves or
rejects. If it never settles, stop admission and drain or terminate the owning
process. Do not replace its admission object or clone its policy to bypass the
charge while the old work is alive. Replica restart must actually terminate the
old owner before replacing its quota. Counters contain no prompt, candidate,
identity, exception, or credentials.

## Wire contract and trust

Version 1 requests contain `{ version: 1, identity, content }`. The identity is a
snapshot of the authenticated execution subject: `runtimeId`, `agentId`, `runId`,
and `userId` are required nonblank strings; `tenantId` and `conversationId` are
optional strings. Omit absent optional fields; `null` is invalid. The host
validates exact equality with its binding. A claimed identity is not proof of
authentication: the transport must authorize that binding before dispatch.

The target is fixed to model-produced message text:
`{ source: 'message', field: 'text', provenance: 'model' }`. Version 1 is not an
arbitrary-target inspection API. Do not infer authorization from content or let
workers redirect the target. Successful replies contain only the canonical
string. The transport must preserve request/response correlation and distinguish
successful strings from failure envelopes.

For failures, encode with `serializeProviderTextProtectionError(error)` and
reconstruct with `deserializeProviderTextProtectionError(envelope)` before
rejecting the worker service call. The versioned envelope is
`{ version: 1, ok: false, error: { code } }`, using the SDK's existing allowlisted
codes. Encoding accepts named SDK errors from separate CJS/ESM module copies;
other exceptions become `unavailable`. Decoding unknown versions, malformed
envelopes, or unknown codes produces `incompatible`. Never send exception
messages, stacks, causes, or raw inspection output. Only decode failures received
from the authenticated host; these helpers are not an authentication mechanism.

Raw candidates travel to the host and must not enter ordinary event journals,
traces, or user-visible persistence. Policy statistics remain host-owned. Custom
graphs must use the SDK release boundary and preserve supplied handlers and
signals; a trusted custom runtime can otherwise bypass protection.

This SDK service is a prerequisite, not complete remote chat parity. LibreChat
still needs an authenticated transport adapter, authoritative builder model
configuration, input/model protections, and the remaining host services before
removing its protected-remote-execution rejection guards.

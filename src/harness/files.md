# Host files and artifacts

`createHostFileBindings` gives an ordinary SDK/custom graph access to host-owned files without
exposing storage credentials, filesystem paths, or signed download URLs. The same versioned
`HostFileService` interface supports direct execution and authenticated remote transports.
`createHostFileService` supplies the host-side identity, validation, resource, and cancellation
boundary. It takes a storage adapter; it does not introduce another file store.

## Contract

| Operation | Meaning                                                   | Host adapter responsibility                                                                                                    |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `read`    | One bounded byte range of an immutable reference/revision | Reauthorize the reference for this identity, check the revision and range, return exactly the requested bytes                  |
| `stage`   | Consume a bounded upload into **private** storage         | Consume through EOF, enforce storage quotas, persist the verified bytes privately, return an immutable descriptor with SHA-256 |
| `publish` | Make an already staged artifact visible                   | Authorize ownership/revision, reconcile the operation ID, pass through native publication, persistence and attachment events   |

The host supplies the initial authorized file descriptors to the graph. `reference` and
`revision` are opaque and retain their exact spelling. A filename is display metadata, not a
storage path. Reads never accept a URL. Browser downloads remain behind the host's normal
authorized download endpoint; there is no transferable download credential in this contract.
Published references can differ from private staging references.

Identity contains the authenticated runtime, root agent, run, user, tenant and optional
conversation. It is snapshotted at binding time and compared before storage access. This is an
assertion against an authenticated host context, **not transport authentication**. Transports must
authenticate the caller, bind to the host-created service, and prevent cross-execution dispatch.
The storage adapter must still check current canonical ACLs and revisions on every operation;
having a descriptor is not an irrevocable grant. Child/sibling grants are not inferred from a
worker's claimed agent ID. A host must explicitly authorize and bind each execution scope.

## Bind a graph

```typescript
const host = createHostFileService({
  identity: authenticatedIdentity,
  descriptor: {
    version: 1,
    operations: ['read', 'stage', 'publish'],
    maxFileBytes: 64 * 1024 * 1024,
    maxChunkBytes: 256 * 1024,
  },
  store: authorizedNativeFileAdapter,
  admission: sharedHostFileAdmission,
  timeoutMs: 60_000,
  signal: executionSignal,
});

const files = createHostFileBindings({
  identity: authenticatedIdentity,
  descriptor: host.descriptor,
  service: host.service, // or an authenticated RPC implementation
});

for await (const bytes of files.read(authorizedAttachment, signal)) {
  await parser.consume(bytes);
}

const artifact = await files.stage(
  {
    operationId: stableUploadId,
    filename: 'report.csv',
    mediaType: 'text/csv',
    bytes: outputSize,
  },
  outputChunks, // AsyncIterable<Uint8Array>, each chunk <= maxChunkBytes
  signal
);
const publication = await files.publish(artifact, stablePublicationId, signal);
```

Keep adapters in closures supplied to graph nodes/tools, not in checkpoint state. Descriptors and
operation IDs are serializable; rebinding after resume must reauthorize them. The conditional
graph in [fileGraph.test.ts](./fileGraph.test.ts) demonstrates read → transform → private staging →
publication over real loopback HTTP and filesystem storage. Its transport and in-memory receipt
map are test fixtures, not a production RPC or durable idempotency implementation.

## Bounds, cancellation and retries

- Share one `HostFileAdmission({ maxConcurrentOperations, maxBufferedBytes })` across bindings in
  the intended process/tenant quota scope. Creating one per RPC defeats aggregate admission.
  Admission reserves two chunks for read/stage validation and snapshot copies. It does not bound
  an adapter's own caches, socket buffers, retained storage, or arbitrary graph state. Those retain
  their existing host limits. Multi-replica quotas require host-level coordination.
- Reads request one range at a time and stop pulling when the graph stops consuming. Range size
  is host-configured; transport round-trip cost must be measured when choosing it. Uploads are
  pulled one chunk at a time, with exact length and host-computed SHA-256 checks. No SDK helper
  materializes a whole file. A backend must stream consumption, not collect chunks into an array.
- The host deadline covers storage work and upload cleanup. Cancellation/timeout returns promptly
  and aborts the adapter signal. Admission remains charged until actual storage work and source
  cleanup settle, including uncooperative work. Store adapters must honor signals and have their
  own bounded I/O deadlines. An execution-level signal also ends work when the RPC has no signal.
- Both stage and publish use stable caller-supplied operation IDs. **The SDK never retries
  mutations or guarantees exactly-once publication.** A timeout/disconnect may occur after a
  commit. The adapter must durably reconcile `(execution identity, operation kind, operationId)`
  across retries/reconnects/replicas, reject a different payload for the same ID, and return the
  canonical receipt without duplicating files or native events. Stage replay still consumes and
  verifies the incoming stream; a stored receipt must not skip byte validation.
- Failed/aborted uploads must be discarded; completed unpublished artifacts require bounded
  retention and cleanup in the native store. Publication must not begin until staging/validation
  completes. A later cancellation cannot undo a completed publication. UI event replay and
  receipt reconciliation belong to the host's existing durable publication pipeline.
- Invalid versions, foreign identities, unsupported operations, excessive sizes, partial streams,
  digest mismatches and mismatched receipts fail closed. Boundary errors expose only fixed codes;
  raw adapter errors, paths, credentials and abort reasons are not returned.

## Integration and parity gates

This SDK addition does not activate LibreChat remote files by itself. The production bridge must
negotiate this capability, encode binary chunks with bounded frames/queues and backpressure, and
connect the store to LibreChat's authorized attachment, private snapshot, publication and download
services. Apply configured file/content protection before bytes leave the host or artifacts become
visible. Preserve native file metadata, previews, event ordering, temporary-chat retention and
deletion. Credentials and policy callbacks stay host-side.

This contract covers attachment bytes and generated file artifacts. Sandbox filesystem/session
mounts, provider-specific multimodal encoding, rich rendered UI artifacts, automatic child grants,
and Console/Schema Advisor product acceptance are separate integrations. The conformance graph
does not prove production authentication, durable restart/recovery, native UI parity or latency
parity. Those are acceptance gates for the LibreChat adapter and product rollout.

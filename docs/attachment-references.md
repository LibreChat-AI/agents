# Attachments by reference

A message can carry an attachment as a reference instead of its bytes. The
reference is LangChain's `source_type: 'id'` data block:

```typescript
{ type: 'file', source_type: 'id', id: 'file-1', mime_type: 'application/pdf' }
```

The host supplies the content when a provider request is prepared, through
`RunConfig.resolveAttachment`:

```typescript
const run = await Run.create({
  runId,
  graphConfig,
  resolveAttachment: (reference, target) => {
    const file = files.get(reference.id);
    if (file == null) return undefined; // not ours: left as it is
    if (!accepts(target.provider, file)) return null; // nothing to send: dropped
    return format(file, target); // the block this provider accepts
  },
});
```

The resolver receives the serving `provider`, its `modelId`, and the
`projectionMode` (`'chat-messages'` or `'openai-responses'`), and returns the
block for that provider. The SDK sends the result as returned.

## What holds the bytes

Resolution happens on the prepared request only. Graph state, checkpoints, and
anything that serializes state hold the reference. The model call receives the
resolved block, so callbacks that observe model input (a trace's generation
span) see it once.

## Contract

- **Synchronous.** The host reads and validates an attachment before the run
  starts, where a failure can still be reported as an attachment error, and
  keeps the bytes for the run. Resolution is then a lookup.
- **Repeatable.** One model call can prepare its request more than once (a
  fallback provider, an overflow retry), each for its own target.
- **`undefined` versus `null`.** `undefined` leaves the block alone, which is
  how provider-side file ids keep working. `null` removes it; a message left
  with no content gets the standard omission notice.

- **Cache breakpoints survive.** An Anthropic `cache_control` marker on a
  reference moves to the block that replaces it, or to the last block that can
  hold one when the reference is dropped.

The resolver reaches every graph of a run: handoff and child agents, and the
summarization request. Remote runs (`createRemoteSdkRun`) do not carry it and
refuse a config that sets one.

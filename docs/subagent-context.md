# Host-owned subagent context

`Run.create({ subagentContext })` lets a host authorize an individual child
execution, supply its initial messages, partition its tool sessions, and project
its successful result. The SDK exports `SUBAGENT_CONTEXT_VERSION = 1` so hosts
can reject an unsupported installation before enabling features that depend on
execution identity.

`prepare(input)` runs before the child graph is constructed or invoked. Its input
contains the SDK-owned `executionContext`, `parentThreadId`, `memberAgentIds`, an
abort `signal`, and a `resumed` marker. The last ancestry entry's `subagentRunId`
identifies this execution; saved agent IDs alone do not distinguish concurrent
copies of an agent. Nested children inherit the adapter and receive their full
ancestry. A graph subagent has one execution identity and multiple member IDs.

Preparation may return:

- `messages`: actual LangChain messages appended after the task description for
  a fresh child. They may contain multimodal file content. Existing checkpoint
  messages are preserved on resume without inserting these messages again.
- `configurable`: host runtime context. SDK run, thread, checkpoint, and execution
  identities cannot be replaced through this object.
- `agentSessions`: entries keyed by child member ID, each replacing that member's
  `codeSessionKey` and `initialSessions`. Omitting `initialSessions` in an entry
  clears inherited session seeds. Unknown members are rejected. A paused live
  graph cannot change its session partition when it resumes.

The SDK calls preparation again when a completed execution is retried and when a
host rebuilds a paused execution, allowing the host to reauthorize access. Make
authorization idempotent for the execution identity. An ordinary preparation
failure prevents child execution and produces a generic failure result. Aborting
the supplied signal instead propagates as an execution error, so callers must
handle it as cancellation rather than as a normal subagent result.

`complete(input, result)` runs after successful child work. It returns the text
delivered to the parent and can append durable file references. The SDK retains
the original completed result if delivery fails, so retrying delivery does not
repeat model or tool side effects. Completion must be idempotent. Failed and
cancelled child work does not invoke completion.

The same canonical `executionContext` reaches child tool hooks,
`ToolExecuteBatchRequest.executionContext`, and the `configurable.executionContext`
and `metadata.executionContext` of direct tool calls. Root tools have no child
context. Event-driven hosts should use the batch field as authority and overwrite
inherited configurable or metadata values before invoking tools and handling
their artifacts.

The adapter does not implement file authorization, storage, publication, or
sandbox isolation. In particular, `codeSessionKey` partitions the SDK's transient
session map; the host must also assign a private runtime workspace and prevent
inherited runtime hints or direct tool implementations from bypassing that
workspace. Published files and any private artifact recovery remain the host's
responsibility. The host must reconstruct its adapter and authorization when resuming. The SDK
does not replay private artifacts to the host when restoring a checkpoint.

## Startup diagnostics

Pass `onSubagentResolutionFailure` to `Run.create` or a graph input. Direct
`SubagentExecutor` callers use `onResolutionFailure`. Detached executors and
nested child graphs inherit the callback.

The synchronous callback receives:

- A read-only diagnostic: `phase` (`identity` or `config`), `subagentType`,
  `aborted`, a closed error `type`, and an SDK-owned `cause` and `message`.
- Correlation IDs: `parentRunId`, `parentAgentId`, `parentToolCallId`, parent
  `threadId`, `childRunId`, `childThreadId`, and detached `taskId` when available.
- The original rejection as a **private second argument**. Use trusted error
  classes or codes to classify it. Never log its message, name, stack, or body.

Return one of `workspace_unavailable`, `agent_unavailable`, `model_unavailable`,
`configuration_changed`, or `unknown`. The SDK builds the parent message from
fixed text; arbitrary runtime return values also fall back to `unknown`.
The input diagnostic starts with `cause: 'unknown'`. Log the mapped cause and
`getSubagentResolutionFailureMessage(cause)` in the host callback:

```ts
onSubagentResolutionFailure: (detail, error) => {
  const cause = classifyStartupError(error); // Host-owned class/code mapping.
  logger.log(detail.aborted ? 'warn' : 'error', 'Subagent resolution failed', {
    ...detail,
    conversationId,
    cause,
    message: getSubagentResolutionFailureMessage(cause),
  });
  return cause;
},
```

A callback returning nothing preserves the unknown fallback. If no callback is
provided, or it throws, the SDK uses `console.warn` with only safe diagnostics.
A callback failure never replaces the child failure.

Direct executor results include `resolutionFailure: { phase, cause }`. Detached
execution throws `SubagentResolutionError` with the same safe `phase`,
`resolutionCause`, and fixed message. Hosts can recognize this class when
mapping background failures instead of exposing arbitrary `Error.message`.
The SDK does not infer host-specific workspace, agent-access, or provider
failures from free text. Those mappings belong to the host.

Graph replay preparation uses the same diagnostic boundary before invoking the
subagent tool. Ordinary preparation failures return a safe error `ToolMessage`
without starting child work. Framework interrupts and stream limits retain their
control-flow semantics. Aborted preparation and settlement identity failures
throw `SubagentResolutionError`; neither exposes the original rejection.

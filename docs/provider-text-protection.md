# Provider-text release boundary (B1 / AI-2214)

Default-off. This SDK supplies an awaited release boundary, not a detector or activation setting.

## Consumer contract

`RunConfig.providerTextProtection`, `StandardGraphInput.providerTextProtection` and direct
`attemptInvoke` / `tryFallbackProviders` accept `ProviderTextProtection`:

- `version: 1`, explicit positive integer `timeoutMs`, `maxAttemptBytes`, `maxBufferedBytes`.
- Trusted synchronous `classify(content)` returns `prose` or `unsupported`. Code, JSON,
  ambiguous mixtures and protocol text must return `unsupported`.
- Required `inspect({ version, content, target, signal })` returns A1's versioned
  canonical-success / raw-free-error result. Target is `message/text/model`.
- Accept/audit returns unchanged content. Redact returns one canonical value. Block returns
  `blocked`. There is no observational-hook fallback.

The same policy object shares its memory budget across concurrent attempts and inherited child
executions. Use one immutable policy object per run. Do not replace it between parallel attempts.
Limits include UTF-16 text, candidate-join capacity and bounded event/attempt overhead, not
only transport bytes. Canonical output is charged too. A producer or handler that ignores
cancellation keeps its lease until it settles; later attempts cannot replenish that budget.
Protected attempts bypass SDK read-ahead smoothing, regardless of configured delay. Only
the gate pulls raw chunks, and an ignoring upstream keeps its lease until actual settlement,
not the smoother close grace. Unprotected smoothing and shared model configuration are unchanged.
The host handler must honor the signal and bound its own detector allocations.

Run-level policy belongs at `RunConfig`, not inside its legacy `graphConfig`; misplaced required
policy is rejected. Custom high-level model/stream overrides are not certified.

Missing handlers, invalid versions/limits/results, deadline exhaustion, buffer overflow and
Stop fail closed. Classification/handler errors become stable SDK error codes, never original
exception details. No failed-attempt prefix is released. Retry/fallback has a fresh candidate.
Policy failures are terminal, not fallback triggers. Queued stream consumers and post-await
eager-tool guards honor captured own and inherited protection signals before dispatch; child-entry and fallback
admission cannot restart work after the trip. Event-tool post-approval admission and child
safety-error pass-through honor the same trips. Foreground child failures abort the captured
parent breaker, not a replacement run. SDK-owned cooperative restart cancellation discards
its attempt without becoming Stop or a policy decision; host cancellation remains terminal. Late decisions cannot release cancelled text. A monotonic elapsed-time check enforces the
deadline even when synchronous policy work prevents the abort timer from running.

## Ordering and supported surface

The boundary clones standard LangChain shells and intercepts `BaseChatModel` generation methods.
Sequences admit only SDK-created, unchanged instruction transforms before one terminal provider.
Multiple-provider sequences, arbitrary prefix callbacks, custom provider transforms/shell overrides and binding
configuration factories fail before any producer starts. Provider prose callbacks are withheld until canonical release. Completed waits remove their
cancellation listeners; every chunk, including empty/control chunks, consumes bounded overhead.
The gate runs **before** LangChain's native aggregation/end callbacks, `attemptInvoke` aggregation,
`ChatModelStreamHandler`, run state and subsequent model/tool reuse. Shared providers are not mutated.

String prose and a single indexed text block are supported. Reasoning, signatures, tool arguments,
IDs, usage and control chunks keep their values and are not buffered behind prose. Anthropic
lifecycle fields, Bedrock block indices/seals and official OpenAI tool-adapter controls are
validated and preserved. Anthropic untyped tool-input fragments must match an admitted
tool block/index and its argument delta; their JSON is never rewritten. Unknown aliases still fail closed. Unknown content
blocks, additional-output aliases, cached models, multiple prose block indices and custom runnable
shells fail `unsupported`. Non-streaming responses are inspected before native callbacks/state;
transport response allocation itself is provider-owned, not an SDK gate buffer. Native nonstreaming
lifecycle/usage metadata and malformed-tool diagnostics are preserved without rewriting.
Invalid calls keep the existing paired-error and model-recovery route after canonicalization. Invoke configured to aggregate internal
streaming, including effective request parameters/model kwargs and OpenAI delegates, fails `unsupported` before production; use the guarded
stream path or native nonstreaming instead.

Native event-stream overrides use the protected chunk bridge. Direct provider use outside
`attemptInvoke` is **not** protected by this run-level contract. Summaries, titles, tool results,
files/media and structured-output mutation are not B1 targets.

## Evidence and ownership

Source baseline: SDK `main@bc5e34b411a500953d4cd745b1439e06780f77a6`, package version `4.0.2`.
A1: LibreChat #16726, merge `d149c495d192f7ae5e734f3129d9a9307be36f88`.
LibreChat's A1 snapshot locks SDK `4.0.1`; it is separate evidence, not this implementation.
`src/protection/__tests__/fixtures/a1.json` is copied verbatim from A1's corpus.

B1 owns `src/protection/providerText.ts` and `src/llm/providerTextBoundary.ts`. C1 owns mandatory
tool-result release and ToolNode/hook ordering. B1 does not change `src/utils/callbacks.ts` or hook
contracts. Shared Run/graph inputs receive only the provider-specific optional field; C1 can add its
own field without changing this handler's semantics.

Automatic instrumentation, provider-internal spans, input/metadata/error export and central/tenant
trace sinks remain D1's independent responsibility. Runtime release is not trace certification.

## Release prerequisite

B2 requires an approved SDK release containing this PR, a check of exported
`PROVIDER_TEXT_PROTECTION_VERSION === 1`, a pin to that release, and its own classification/policy adapter. Old SDK + required consumer and
required SDK + uncertified consumer must stay gated. No version bump, npm publication or app/YAML
activation is included. Drain protected attempts before rollback; never retry them without policy.

# Domain Context

## Durable Subagent Execution

A **Durable Subagent Execution** is one child invocation identified by its
durable parent thread, parent checkpoint fork, parent agent, parent tool call,
tool batch, and resume attempt. Concurrent parent forks and resume attempts are
different executions even when they reuse the same parent tool-call ID.

An **Execution Record** is the in-process source of truth for one Durable
Subagent Execution. It owns the canonical execution address, resolved child
run and checkpoint-thread identity, approval scope, effective subagent type and
configuration revision, pending resolution and invocation work, active graph
state, completion state, and invalidation.

Child identity resolution is prepared before it is committed. Checkpoint
forks, approval replay restoration, and cleanup tracking become authoritative
only while the same Execution Record is still current; invalidated
preparations roll back only resources owned by their preparation lease. A
checkpoint writer keeps exclusive ownership until its preparation settles and
any owned cleanup finishes, so a retry cannot adopt a branch that the old
writer can still modify. Pre-existing branches are never deleted as
preparation-owned state. Active and cleaning resources stay fenced; retries
fail closed and can reacquire them only after they become free.

An **Invocation Binding** is the immutable subagent type, configuration
revision, and description accepted when an Execution Record starts. Exact
duplicate dispatches share its pending result. A same-address dispatch with a
different invocation fails closed instead of observing the first dispatch's
result.

After a resume attempt successfully forks its source checkpoints, the source
Execution Record is invalidated and removed. Its active graph, completed
messages, resolved configuration, and approval session are retired together.

An **Effective Definition Binding** is the subagent type and optional
configuration revision that actually ran after lifecycle hooks. Durable replay
must use this binding rather than reconstructing it from the original tool
call.

A **Resume Projection** is the durable manifest view produced from an
Execution Record. A projection is scoped to one exact fork and resume attempt;
an unscoped projection fails closed when a parent tool-call ID is ambiguous.

## Event Actors

An **Event Actor** is one stable logical child thread that handles a sequence
of authoritative host events without retaining a live executor between them.

An **Event Actor Head** is the current committed checkpoint identity and its
monotonic generation. Each invocation runs on an **Invocation Fork** owned by
that invocation. Once committed state exists, every fork stays on the same
logical checkpoint thread and changes only its invocation namespace. The host
advances the Event Actor Head only through an atomic comparison against both
the generation and prior checkpoint identity.

An **Event Actor Event** is immutable JSON data snapshotted at every public and
host-adapter boundary; signed zero normalizes to JSON's zero representation.
Checkpoint snapshots retain only the declared identity fields. Cold
reconstruction receives the explicit task-owned cancellation signal and owns
rollback until it returns a validated invocation.
Prepared invocations and unavailable request/head pairs carry canonical,
time-bounded, executor-authenticated integrity bindings so independently valid
lifecycle evidence cannot be forged, recombined, or replayed after expiry.
Cold continuation consumes its unavailable handoff before adapter work. Hosts
that persist those handoffs across executor lifetimes provide the same private
preparation signing key to every authorized executor.
Public invocation produces an executor-issued **Applied Settlement** that binds
immutable result and terminal checkpoint evidence to the exact invocation
reference that ran. A **Suspended Event Actor Invocation** is authenticated,
versioned, JSON-safe evidence that the same logical invocation remains
nonterminal at one exact paused checkpoint and interrupt. It contains no graph,
configuration, signal, closure, or process-local handle. **Suspension
Authority** is the combination of that integrity evidence and the host's
durable current-state fence: the signature prevents forgery and recombination,
while the host fence prevents replay and proves one-shot ownership.

A **Resume Attempt** is one host-claimed decision identified by a unique attempt
ID. The host must claim it before applying the decision and must reject every
competing, duplicate, stale, expired, or foreign claim. When resumed execution
pauses again, **Re-pause Replacement** atomically publishes a new suspension
with a new identity and monotonic attempt while consuming the claimed old
suspension. Neither suspension is current if that comparison-and-swap cannot
be proven. A resumed applied result carries signed settlement authority; the
host consumes its suspension fence atomically with the actor-head comparison,
so settlement may move to another authorized executor without process-local
object identity. Cancellation similarly claims, discards, and closes the
current suspension as one host operation. Expiry removes signature authority
to resume but never proves an external action outcome; cleanup of expired
evidence still requires a host-proven current-state transition.

Suspension and resumed-settlement versions are fail-closed compatibility
boundaries. During a rolling deployment, the host must capability-route durable
records only to executors that support their exact version; a terminal-only old
executor must never consume a new suspended job. Unknown versions are rejected
before host mutation. Every executor that may handle the same current evidence
uses the same signing key. Key rotation therefore requires draining, cancelling,
or explicitly migrating all current signed evidence before the old key is
removed; silently replacing the key strands valid paused work by design.

Invocation start phase-fences the prepared capability so it cannot subsequently
authorize discard of active or applied work. Commit consumes local settlement
identity or authenticated resumed settlement authority and returns structured
indeterminate evidence for missing or invalid acknowledgements rather than
exposing a retryable-looking exception. A host mailbox remains the durable
cross-runtime owner and deduplication fence. Public invocation reclaims
definitely-no-action forks before returning or rethrowing, while ambiguous and
applied outcomes stay retained. The executor opportunistically prunes local
terminal phase fences after the dormant-checkpoint TTL without scheduling
timers that retain short-lived executors. A fence never expires before its
signed authority, so pruning cannot re-enable a stale capability; the mailbox
prevents stale cross-runtime replay after expiry.

Applied work may commit its Invocation Fork. Failed, cancelled, and
completed-without-action invocations discard their forks. An applied fork that
loses the head comparison remains available for reconciliation. A missing or
incompatible warm checkpoint uses **Cold Continuation**: the host rebuilds an
Invocation Fork from bounded transcript and summary state while preserving the
same logical Event Actor identity. An indeterminate applied settlement,
including malformed action checkpoint evidence or an ambiguous commit
acknowledgement, is retained for reconciliation because deleting its fork
could erase the only durable evidence of an external action.
Hosts may translate a durable LangGraph pause into the structured suspension
result. Untranslated LangGraph interrupts and parent commands retain the fork
and propagate as control flow for compatibility and routing.

## Terminal Run Continuation

A **Terminal Run Continuation** keeps one naturally completed `Run` alive when
its terminal hooks admit queued host input. Ordinary Stop hooks execute in
parallel and fold policy/notification requests first; the serialized
StopFinalize phase then observes whether they already planned or prevented a
continuation, allowing a durable host to make one atomic claim-or-seal decision.
A `block` result with injected messages starts another graph segment before Run
cleanup, hook-session teardown, or final content extraction. The continuation
retains the same public run, trace scope, event handlers, graph sidecars, and
content accumulator; it does not replay RunStart or UserPromptSubmit hooks.

Continuation is bounded. Both terminal phases report the admitted count and
remaining budget. StopFinalize additionally reports the folded planned and
prevented state so unrelated continuation sources cannot close each other's
admission. HITL pauses, aborts, hook halts, output truncation, and empty
Stop injections remain terminal and never enter this path. With a checkpointer,
the next segment submits only its injected delta; without one, it seeds the new
graph invocation from the live in-process transcript.

## Tool Caller Capabilities

A **Caller Capability Projection** is the effective classification of tool
definitions by the contexts permitted by `allowed_callers`: direct model calls,
programmatic code execution, both, or neither. Prompt guidance, model binding,
and runtime dispatch derive from this projection rather than recomputing caller
rules independently.

A **Programmatic Tool Manifest** is the exact list of registered tool names a
programmatic invocation declares that its code depends on. Mixed direct and
code-execution configurations require the manifest so caller policy can reject
direct-only dependencies before starting an execution runtime without parsing
the submitted programming language.

## Context Pressure Measurement

A **Context Pressure Meter** is the request-scoped projection of retained
conversation messages into the provider's available context budget. It keeps
provider-grounded attribution and exact token counts together so repeated
artifact, compaction, prepared-request, and fallback probes tokenize only new
message objects. Its Agent Context may retain exact counts across provider
requests only for stable string-content messages whose token-relevant surface
still matches.
Mutable or ambiguous message shapes are recounted. Provider projections must
clone changed messages; mutating a measured message invalidates its retained
count on the next comparison. Custom token counters remain uncached unless a
host explicitly declares that they implement the same deterministic surface
contract.
See [ADR 0001](docs/adr/0001-reuse-exact-context-token-counts.md) for the
cache boundary and host-lifecycle trade-offs.
Host-supplied index token counts are the baseline's accounting weight; a
retained message is tokenized only when no such count exists or when a
provider projection replaced its object, since the exact count then serves
only as the subtrahend of that projection's delta. Native brand checks in
content validation use `node:util/types` slot reads rather than throwing
probes, so plain content objects never pay an exception per value.

Provider SDK modules load on first use, never at import time: built-ins register
loaders with the provider registry, `instanceof` guards resolve their classes
through the same lazy seam, and every such load goes through `src/lazyRequire.ts`
— the one module allowed to touch `import.meta`, stubbed in jest so tests resolve
source modules. Adding an eager import of a provider SDK anywhere on the root
entry's graph regresses every host's boot.

The meter's estimates decide overflow and recovery only. Provider-reported
usage remains the source of truth for billing and Langfuse observations.

A **Compaction Range** is the contiguous conversation prefix replaced by one
durable checkpoint. Range selection prefers whole user-led turns. If that
would leave no compactable prefix in a tool-heavy run, it may cut after an
older closed tool-call/result unit while retaining a token-priced recent tail.
Open and parallel tool units are indivisible, and a lone user payload is never
eligible for the intra-turn fallback.

A **Cache-Aligned Compaction Request** reuses the normal provider request's
cacheable tool prefix and anchors retained history before the one-off
summarization instruction. Provider-specific cache markers and TTL resolution
come from shared request-preparation functions; cache-read usage is the source
of truth for whether reuse occurred. Compaction never predicts cache identity
from a hand-maintained list of provider options or environment variables.

A **Compaction Semantic Index** is optional navigation guidance derived during
Model Context Reconstruction's existing persisted-content analysis. Its
committed, user-visible entries are source-addressed, revisioned,
redaction-aware, deterministically ordered, and strictly bounded. The host
owns enablement and identifies which tool `intent` fields are semantic labels.
The index is appended only after cacheable raw history; it never replaces raw
messages or makes generated labels authoritative.

An **Incremental Compaction Semantic Index** evolves a previously committed,
bounded index with labels from only the current persisted payload. Its
serializable snapshot contains retained entries plus their cumulative supplied
count so omission telemetry survives process boundaries. Model Context
Reconstruction snapshots the prior entries, admits them chronologically through
the same coverage-balanced collector, and derives the delta during its existing
content pass. Work is independent of unbounded conversation length: `O(B +
delta)`, where `B` is capped at 256. Invalid prior state degrades to delta-only
derivation rather than disabling new guidance. Ephemeral revision floors for
the bounded base prevent coverage omissions from admitting stale delta labels;
they never become serialized collector state.

See [ADR 0005](docs/adr/0005-pairing-balanced-compaction-ranges.md) for the
range-selection and retention trade-offs.
See [ADR 0007](docs/adr/0007-guide-compaction-with-a-bounded-semantic-index.md)
for the index trust, ordering, and latency boundaries.
See [ADR 0008](docs/adr/0008-evolve-compaction-guidance-from-warm-turn-deltas.md)
for bounded warm-turn evolution and continuation ownership.
See [ADR 0006](docs/adr/0006-align-compaction-with-provider-cache-prefixes.md)
for the prompt-cache boundary and rejected predictive replay design.

## Provider Tool Derivation

A **Provider Tool Derivation** is the copy-on-write projection of retained
assistant history that drops incomplete streamed text blocks and bounds every
provider-consumed tool-call input representation in one pass. Its output is
the provider-safe preflight input for context-pressure measurement; provider-
specific replay and wire shaping remain later request projections.

## Model Context Reconstruction

An **Ordered Tool History Projection** preserves the contributions carried by
provider history together with their order, actor, item identity, and text or
media payload. Complete Responses output and partial streaming sidecars have
different coverage: a sidecar cannot replace ordinary message content or prove
its position without recorded replay positions. Folding and replay share source
selection and generated-image interpretation. Native replay policy remains in
request preparation; the projection never mutates checkpoint messages.

A **Session Log** is the append-only source of persisted conversation events.
It retains message, summary, and lifecycle history without becoming a second
mutable message state.

A **Session Message Projection** is the deterministic reconstruction of model
context from a Session Log. Summary events replace earlier projected context;
message events append in log order. Agent runs, compaction, and resume derive
their starting messages through the same projection.

A **Provider Message Projection Invariant** checks the final message batch at
the chat-model callback boundary, after runnable-level instruction injection
and before provider serialization or I/O. Source-backed user, model, and tool
contributions require provenance; source-less contributions are valid only
when explicitly synthetic. The check reports counts, message positions, and
roles only—never content, source IDs, or tool data.

The invariant is opt-in through `AGENT_MESSAGE_PROJECTION_INVARIANT=observe`
or `assert`. The default `off` path adds no callback handler and performs no
message scan. The initial rollout uses observation to locate provenance gaps;
assertion is reserved for tests and environments whose projection is complete.

## Runtime Provider Registration

A **Provider Registration** is the process-local binding of one provider name
to its chat-model constructor and the message/streaming traits the agents
runtime must apply. Built-in and host providers resolve through the same
registry. Hosts register before constructing agent configuration; duplicate
names fail closed, and a disposer removes only the exact registration that
created it.

Provider family, manual tool-stream handling, and strict message alternation
belong to the registration. Family governs shared protocol behavior such as
thinking detection, output-token option names, and atomic tool-call handling.
Exact provider-specific behavior such as Bedrock cache shaping or OpenRouter
replay handling remains keyed to the built-in provider identity. Host option
types are declaration-merged through the stable
`@librechat/agents/provider-registration` subpath without widening the built-in
option maps.

See [ADR 0003](docs/adr/0003-runtime-provider-registration.md) for the shared
registry seam and process-lifecycle trade-offs.

## Coding-Tool Execution

An **Execution World** is the stable pairing of a filesystem namespace and a
subprocess launcher used by coding tools. The Node host has one default world;
remote backends retain one world per execution configuration so rebuilding an
agent's tool binding does not make a warm backend look new to capability
probes. Sandbox identity belongs to the same world, while timeout,
cancellation, output, and command results remain invocation-scoped.

See [ADR 0002](docs/adr/0002-stable-execution-worlds.md) for the adapter
identity and compatibility boundary.

## Prepared Subagent Invocations

A **Prepared Subagent Invocation** is a foreground built-in subagent invocation
started after its provider seals the arguments but before the parent model
attempt completes. It is owned by that attempt and the executing agent, bound
to the tool-call ID and canonical arguments, and adopted once by ToolNode for
normal result processing. It is process-local and is not a Durable Subagent
Execution replay mechanism. See [ADR 0009](docs/adr/0009-prepare-sealed-subagent-invocations.md).

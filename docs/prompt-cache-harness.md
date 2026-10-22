# Prompt-cache harness

An offline diagnostic suite that checks whether consecutive requests of a
conversation can reuse a provider's prompt cache. It builds every turn of a
scenario through a real `Run` and the real provider chat model class, captures
the serialized request right before it would leave the process, aborts it, and
asserts byte-level invariants on what was captured. No request reaches the
network, so it runs in the default `npx jest` and in CI.

- Harness: [`src/specs/prompt-cache/harness.ts`](../src/specs/prompt-cache/harness.ts)
- Provider cases: [`src/specs/prompt-cache/providers.ts`](../src/specs/prompt-cache/providers.ts)
- Scenarios: [`src/specs/prompt-cache/scenarios.ts`](../src/specs/prompt-cache/scenarios.ts)
- Invariants: [`src/specs/prompt-cache/prompt-cache.test.ts`](../src/specs/prompt-cache/prompt-cache.test.ts)

```bash
npx jest src/specs/prompt-cache
```

## What is compared

Each captured request is reduced to the order providers cache in — tools,
system, messages — and serialized one JSON segment per line. Before
comparison, cache markers (`cache_control`, `cachePoint`) are removed and
string content is compared in single-text-block form: markers move to the
newest turn by design and are not prompt content, and marking a message turns
its string content into a text block that providers read as the same prompt.

Invariants, for every provider case and every scenario:

| invariant | meaning |
| --- | --- |
| conversation block in system | `conversation_instructions` is always part of the system prompt |
| only the per-turn part moves | `additional_instructions` travels in the messages exactly for the cases that declare `movesDynamicPart`, and stays in system otherwise |
| tools and system fixed between reset points | apart from the per-turn part, nothing above the history changes while the conversation context is unchanged |
| request N+1 starts with request N | up to the message that carries the previous turn's per-turn part, or entirely when there is none; a follow-up call within a turn extends the previous call entirely |
| prefix kept up to the boundary at reset points | a declared change (new `additionalInstructions`, host compaction) keeps at least the stable instructions; when compaction appends tools, at least every tool that was already there (tools are appended, never re-sorted) |
| shared prefix across conversations | two conversations of the same agent share tools and stable instructions |
| breakpoint limits | at most four `cache_control` (Anthropic, OpenRouter) or `cachePoint` (Bedrock) markers per request |
| strict alternation | no two user messages in a row where the provider requires alternation |

## Adding a scenario

Append a `Scenario` to `SCENARIOS`. A scenario has stable `instructions`, a
tool list, and one or more conversations; each conversation has its own
`conversationInstructions` and a list of turns:

- `user` / `answer` — the exchange; the answer joins the history after the turn
  is captured.
- `toolRounds` — tool calls and results the assistant made before `answer`;
  the follow-up model call after every round is captured as well.
- `additionalInstructions` — per-turn volatile context for this turn only.
- `conversationInstructions` — replaces the conversation context from this
  turn on. Declares a reset point.
- `compaction` — host-side compaction before this turn: `summary` becomes the
  run's initial summary, history restarts at `retainFromTurn`, `appendTools`
  join the end of the tool list. Declares a reset point. With `afterRound`
  the compaction happens inside this turn's run after that many tool rounds:
  the next request is the reset point, and the following requests and turns
  (restored with the summary as `initialSummary`) must continue it.

Every invariant applies to the new scenario automatically.

Turns before the current one are not built as messages directly: the harness
stores them as the content parts the SDK aggregator produces and restores them
with `formatAgentMessages`, as a host does. Each `toolRounds` entry is a
separate model response (optionally with `text` before its call); the
follow-up call after every round is captured.

## Restored history

What a host persists decides whether a restored turn serializes to the bytes
the live run sent. The aggregator writes it onto the content parts; hosts that
store the parts unchanged need nothing else.

| stored on | field | needed for |
| --- | --- | --- |
| `tool_call` part | `tool_call.roundId` | keeps each model response's calls in its own assistant message; without it sequential rounds merge into one parallel batch, and only text between calls separates them |
| `tool_call` part | `tool_call.providerItems` | OpenAI Responses: the response's output items up to and including this call (`{ origin, created_at, items }`). Restored verbatim as the assistant message's provider output, so ids, status, reasoning items and assistant message items match the live request |

| `tool_call` part | `tool_call.outputFormat` | `blocks` when an in-process tool returned content blocks: `output` holds their JSON and is restored as the same blocks, not as one string |

`providerItems` arrive after the model response completes, through the
registry-only `GraphEvents.ON_PROVIDER_OUTPUT_ITEMS` event: a host routes it
to `aggregateContent` like the other step events (it carries encrypted
reasoning, so there is no need to forward it to a client). The items are
replayed only to the model and endpoint that produced them (`origin` is a
digest of both), never with `zdrEnabled`, and a reasoning item without
`encrypted_content` is left out once it is older than the 30-day retention
of stored responses, since it is only a reference to that stored copy. A
round restored without items falls back to rebuilding the items from the
message fields, without an empty assistant message.

### Restored equals live

[`liveRestore.test.ts`](../src/specs/prompt-cache/liveRestore.test.ts) is the
strongest invariant and is generic over providers: every provider case runs
real turns against a scripted stream of its own wire protocol
([`scriptedReplies.ts`](../src/specs/prompt-cache/scriptedReplies.ts): Chat
Completions, Responses, Anthropic Messages, Gemini, Bedrock Converse), stores
the aggregator's parts, restores them, and requires the first request of turn
N+1 to start, segment by segment and byte for byte, with the last request of
turn N. Only cache markers are set aside. The scenario mixes parallel and
sequential rounds, text before calls, an empty text before a call, tool
results that are plain text, several text blocks and an image, for both
in-process and event-driven tools. Any new shape drift between a live and a
restored round fails it for the provider concerned.

Shape rules this relies on: an Anthropic `tool_result` whose content is one
plain text block is sent as its string (the tail cache marker otherwise
turned the string into that block for one request only), and with OpenRouter
prompt caching every non-empty tool message is sent as one text block (the
breakpoint needs a block).

Reasoning: signed `thinking` (including empty text) and `redacted_thinking`
reach the follow-up request of their own turn unchanged, and Bedrock signed
`reasoning_content` likewise; the suite checks this as a regression. Without
the opt-in below, restored history does not replay reasoning of earlier turns,
so the turn boundary is compared with those blocks set aside. Unsigned
`thinking` (as some proxies emit) is dropped on both paths, as before,
because the Anthropic API rejects it. None of this was verified against the
real Anthropic or Bedrock APIs; the suite drives scripted streams.

#### Replaying signed reasoning (opt-in)

Models that keep earlier turns' thinking (a signed block binds the model and
the conversation before it) need those blocks back in the restored history, or
the cached prefix ends at the first thinking block of the previous turn. A
host opts in on both sides: `AgentInputs.providerReasoningReplay` makes the
aggregator store the blocks on the `think` part (`provider_replay`, with the
model that served the call), and `formatAgentMessages(..., {
replayProviderReasoning: { provider, model } })` puts them back, at the slot of
their part, only for that provider and model. Anthropic requests of an agent
with the opt-in also carry `thinking.block_binding.prefix_mismatch_behavior:
"drop_block"` and its beta header, so a block that no longer matches is dropped
by the API instead of failing the call; the response's `input_transformations`
is kept in `response_metadata`.

A signed block is valid only if everything before it is unchanged, so replay
starts after the last hole: any assistant message with reasoning that cannot be
replayed (not stored, edited away, over a host cap, signed by another model)
and the message that carries a summary drop every block before them, and only
the unbroken run after it is sent. Bedrock cannot drop a mismatching block, so
an agent with the opt-in retries a request once without signed reasoning when
Bedrock rejects it for that (one warning).

The `... replay` cases of `liveRestore.test.ts` (`keepsPriorThinking`) compare
the turn boundary with the reasoning blocks included: signed, redacted, empty
text, thinking after the text of a response, Anthropic and Bedrock, through
the dispatcher too. Each scripted reply has its own signatures, the replayed
history must be append-only (no thinking block disappears from the middle),
and the same scenario without the opt-in must break the boundary, which
catches a reverted replay. `reasoningReplay.test.ts` covers a model change
between turns and a call served by a fallback model.

[`restoredToolRounds.test.ts`](../src/specs/prompt-cache/restoredToolRounds.test.ts)
and [`responsesToolRounds.test.ts`](../src/specs/prompt-cache/responsesToolRounds.test.ts)
check this end to end: real runs against scripted Chat Completions and
Responses streams (in-process and event-driven tools, reasoning encrypted,
by reference and off), the aggregator's parts, and every next request
starting with the whole previous one.

## Adding a provider

Append a `ProviderCase` to `PROVIDER_CASES`:

- `provider` and `clientOptions` — as a host would pass them; use fake keys and
  `maxRetries: 0`.
- `intercept` — install the capture point and return a restore function.
  Clients that call `fetch` use `fetchCase`, which spies on the global `fetch`.
  Clients with an injectable transport get it through `clientOptions` (Bedrock
  receives a `client` whose `send` records `command.input`).
- `reduce` — map the captured body to `{ tools, system, messages }` in cache
  order. Reuse `reduceChatCompletions`, `reduceResponses`, `reduceAnthropic`,
  `reduceGoogle` or `reduceBedrock` when the wire format matches.
- `breakpoints`, `movesDynamicPart`, `strictAlternation` — declare the
  provider's cache markers and limits, whether it moves
  `additional_instructions` into the messages, and whether it requires
  alternation.

Not covered: Mistral (its client is ESM-only and stubbed under Jest, see
`test/stubs/mistralai.ts`; it shares the graph's alternation path with
Bedrock) and Vertex AI (needs Google auth before the request is built).

## Instruction tiers

An agent's system prompt has three tiers, ordered from most to least stable:

| field | lifetime | Anthropic / OpenRouter with `promptCache` | Bedrock with `promptCache` | other providers |
| --- | --- | --- | --- | --- |
| `instructions` | every conversation of the agent | system block with a breakpoint | system block + cache point | system text |
| `conversation_instructions` | one conversation, changes only at its start | next system block with a breakpoint | next system block + cache point | appended to the system text |
| `additional_instructions` | one turn | user message ahead of the newest human message | after the cache points in system | appended to the system text |

The summary (a compaction in the run or `initialSummary`) is a separate user message right after the system message, stable until the next compaction, with a breakpoint of its own where one is left. The conversation block is never moved, so within a conversation each request
extends the previous one up to the per-turn part, and `instructions` stays
shared across conversations. When the messages already carry two breakpoints
(stable history and tail), the conversation block's own breakpoint is dropped
to stay within Anthropic's limit of four; the stable-history breakpoint after
it covers it.

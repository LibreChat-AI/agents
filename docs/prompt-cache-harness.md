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
- Tool dispatch: [`src/specs/prompt-cache/tool-dispatch.test.ts`](../src/specs/prompt-cache/tool-dispatch.test.ts)

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

## Tool dispatch

`tool-dispatch.test.ts` makes a tool available mid-conversation through the
dispatcher (`AgentInputs.toolDispatch`), for every provider case, with in-graph
tools and with event-driven tools executed by a host handler, in two ways: the
host announces it and the model calls `call_tool`, or the model finds it with
`tool_search` and calls it directly (the SDK rewrites that call into a
dispatcher call). Model replies are streamed tool calls faked in the provider's
own wire format (`FakeToolCall`; a provider case's `intercept` answers a
request when its `record` callback returns one). The turn after the call is rebuilt from the
content parts a host stores, through `formatAgentMessages`. Checked: `tools`
is byte-identical on every request and never lists the target, each request
starts with the previous one, the target runs once with its own arguments, the
result answers the dispatcher call, and history names only declared tools.

Known gaps, independent of the dispatcher, at the turn rebuilt from stored
parts: the Responses API replays a live tool round as the provider's output
items (ids, status), while a rebuilt round has neither and gains an empty
assistant message; consecutive tool rounds without text between them are
rebuilt as one parallel batch. There the next turn reuses the prefix only up to
the tool rounds.

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

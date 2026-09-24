# @librechat/agents

TypeScript utilities for building LibreChat agent workflows. The package provides graph orchestration, streaming event handling, tool execution, provider adapters, and message formatting for single-agent and multi-agent runs.

## Features

- LangGraph-based single-agent and multi-agent workflows
- Streaming content aggregation and run-step event handlers
- Tool calling, tool search, subagent handoffs, and programmatic tool execution
- Provider adapters for Anthropic, Bedrock, Vertex AI, OpenAI-compatible providers, Google, Mistral, DeepSeek, and xAI
- Message formatting, context pruning, summarization, and cache-control helpers

## Installation

```bash
npm install @librechat/agents
```

## Basic Usage

```typescript
import { HumanMessage } from '@langchain/core/messages';
import { Providers, Run } from '@librechat/agents';

const run = await Run.create({
  runId: crypto.randomUUID(),
  graphConfig: {
    type: 'standard',
    instructions: 'You are a helpful assistant.',
    llmConfig: {
      provider: Providers.OPENAI,
      model: 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
    },
  },
  returnContent: true,
});

const content = await run.processStream(
  { messages: [new HumanMessage('Hello')] },
  {
    runId: crypto.randomUUID(),
    streamMode: 'values',
    version: 'v2',
  }
);
```

## Runtime Provider Registration

Hosts can register a compatible LangChain chat model before creating a run,
without adding the provider to this package. The provider registration carries
the model constructor plus the shared message and streaming behavior the model
needs.

```typescript
import { registerProvider } from '@librechat/agents/provider-registration';
import type { LLMConfig } from '@librechat/agents';

interface AcmeOptions {
  apiKey: string;
  model: string;
}

declare module '@librechat/agents/provider-registration' {
  interface CustomProviderOptionsMap {
    acme: AcmeOptions;
  }
}

const unregister = registerProvider({
  provider: 'acme',
  model: AcmeChatModel,
  family: 'openai',
});

const llmConfig: LLMConfig = {
  provider: 'acme',
  apiKey: process.env.ACME_API_KEY!,
  model: 'acme-chat',
};
```

Register once per process before the provider is used. Duplicate names are
rejected, and `unregister()` removes only that registration. Set
`manualToolStream` or `strictAlternation` only when the provider contract needs
those behaviors. Declaration-merged required options are enforced for direct
model initialization, graph agents, primary graph configs, and fallbacks.

## Programmatic Sessions

For scripts, CI, and programmatic integrations, use the session facade. It
keeps a JSONL session tree by default, so runs can be resumed, cloned, forked,
branched in place, compacted, and inspected later.

```typescript
import { Providers, createAgentSession } from '@librechat/agents';

const session = await createAgentSession({
  checkpointing: true,
  graphConfig: {
    type: 'standard',
    instructions: 'You are a concise coding assistant.',
    llmConfig: {
      provider: Providers.OPENAI,
      model: 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
    },
  },
});

const result = await session.run('Summarize this repository.');
console.log(result.text);
console.log(session.sessionPath); // durable .jsonl session file
```

When `checkpointing` is enabled, the session injects a shared LangGraph
checkpointer into `compileOptions`, records checkpoint IDs in JSONL, and uses
checkpoint state for later turns on the same `thread_id`. When HITL is enabled
(`humanInTheLoop: { enabled: true }`), sessions also get a `MemorySaver` by
default so `resumeInterrupt()` can reuse the same saver instead of relying on a
per-run fallback. JSONL still owns portable replay, clone, fork, and audit
records.

Sessions expose tree operations inspired by Pi-style workflows:

```typescript
const store = session.getSessionStore();
const forkPoint = store?.getForkPoints()[0];

if (forkPoint) {
  const forked = await session.fork(forkPoint.id, { position: 'before' });
  await forked.run('Try a different approach from here.');
}

const cloned = await session.clone();
await cloned.compact({ instructions: 'Keep only implementation decisions.' });
```

`session.stream()` projects the SDK's existing graph events, and
`session.compact()` uses the same summarization node, hooks, and provider
logic as normal runs. JSONL is the durable journal; the graph remains the
execution engine.

OpenAI-compatible streaming helpers are available as experimental subpaths:

```typescript
import { composeEventHandlers } from '@librechat/agents';
import { createOpenAIHandlers } from '@librechat/agents/openai';
import { createResponsesEventHandlers } from '@librechat/agents/responses';

const customHandlers = composeEventHandlers(
  createOpenAIHandlers(openAIConfig),
  createResponsesEventHandlers(responsesConfig),
  hostHandlers
);
```

For hosts that need stable OpenAI-compatible tool calls across graph invocations,
`createOpenAIToolCallStream` is an **opt-in** accepted-result projector. Pass its
`handlers` to `Run.create({ customHandlers })` (compose with other host handlers
using `composeEventHandlers`). Streaming hosts pass `{ tracker, emit }`, sharing
`createOpenAIStreamTracker()` with `sendOpenAIFinalChunk`. The projector emits the
initial assistant role if needed. Terminal state follows accepted model-response
order, not the deferred tool-chunk flush: a tool → tool-result → final-answer run
ends with `stop`, even though buffered tool history is emitted last. A last
accepted response requesting tools ends with `tool_calls`. Assistant text emitted
after projection can still change the finish reason back to `stop`.
Map-only `{ toolCalls }` is for non-streaming collection or custom finalization;
never compose the legacy raw tool-call handlers with this accepted-result handler,
which would publish the same tools twice. The graph delivers `ON_MODEL_RESPONSE`
once per accepted AI result, after primary/fallback selection, overflow recovery
and usage accounting. Both streaming and invoke-only paths use this boundary.

This event carries detached, finalized tool calls. It is not a provider callback,
a UI run step, or a parseable JSON fragment. The projector never reconstructs raw
chunks or infers attempt identity. Generic callback echoes of the event are
ignored. The existing `createOpenAIHandlers` and `/responses` outputs are unchanged.

Call `finish()` only after confirming **natural run completion**: `processStream`
must not have thrown, the caller/graph signals must not be aborted, and
`getInterrupt()` / `getHaltReason()` must be empty. A resolved `processStream()`
alone does not promise successful completion. On failure, disconnect, halt or
interrupt, call `abort()` and discard this projector. Checkpoint resume requires a
fresh projector; this helper is not a durable delivery/replay protocol. A new
projector is required for each API response, including reuse of a `Run` instance.

Each projector must own a fresh, empty output map (including a tracker's map).
Reusing nonempty output is rejected without clearing the old response. A second
writer during collection aborts projection rather than mixing response histories.

The accepted-result graph boundary inspects original tool-call descriptors and
validates argument trees **before** cloning can invoke getters or flatten class
instances. The graph snapshot is capped at 1,024 calls and 4 MiB per accepted
model response. Invalid-tool diagnostics are rejected before copying their
contents; malformed/accessor-backed tool arrays cannot bypass snapshot limits. Composed observers receive independent copies of that validated
snapshot, so their ordering cannot change what the projector publishes or the
arguments tools execute. Copy work is linear in the number of observers and
bounded snapshot bytes. Synthetic IDs are assigned only at completion, after
all provider IDs have been reserved, and their bytes count toward the projector's
limit.

At acceptance, tool arguments are validated and copied into a detached graph
snapshot; the protocol formatter subsequently encodes them for output and
buffers only completed calls until successful run completion. Argument
trees must contain only JSON primitives, plain objects and dense arrays. Map,
Set, Date, RegExp, boxed primitives, typed arrays, custom instances, proxies,
accessors, symbol properties and sparse arrays are rejected rather than silently
changing the accepted values. Encoding does not invoke `toJSON` or getters and
preserves negative zero. Plain objects from another realm and null-prototype
objects are supported. Nesting is capped at 64 levels, and cycles are rejected.
Repeated references serialize by value, with every expansion charged to the byte
budget so small alias graphs cannot generate unbounded output. Default
limits are **1024 tool calls** and **4 MiB of serialized argument, name and ID
UTF-8 bytes**, configurable with positive `maxToolCalls` / `maxBufferedBytes`.
Exceeding a limit aborts projection without publishing a partial batch. There are
no fragment buffers or per-attempt maps. These limits bound retained projection
state and encoder output, not provider buffers or the earlier graph-owned
`structuredClone` of an accepted model result. The encoder itself checks size while
traversing, before allocating an unbounded serialized string. They do not limit
agent execution globally.

The emitter remains synchronous. Errors or cancellation during emission are
terminal; bytes already sent cannot be retracted and `finish()` cannot retry a
partial write. The host owns HTTP connection lifecycle and async backpressure.
Tool execution, including existing eager execution, is unchanged: projection
failure cannot roll back tool side effects. No provider fallback is initiated by
an acceptance-handler error.

## Development

```bash
npm ci
npm run build
npm test
npx tsc --noEmit
npx eslint src/
```

## Documentation

- [Multi-agent patterns](./docs/multi-agent-patterns.md)
- [Summarization behavior](./docs/summarization-behavior.md)

## License

MIT

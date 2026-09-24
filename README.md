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

### Accepted tool-call projection (opt-in)

`createOpenAIToolCallStream` from `@librechat/agents/openai` formats finalized
calls accepted by the graph, not provider fragments. Existing handlers are unchanged.

- Register `projection.handlers` in `Run.create({ customHandlers })`.
- For streaming, share `{ tracker, emit }` with the text/reasoning handlers and
  `sendOpenAIFinalChunk`. Do not also register legacy raw tool-call handlers.
  Map-only `{ toolCalls }` supports JSON/custom finalization. Use fresh output state.
- Call `finish()` only after natural completion: no error, aborted signal,
  `getInterrupt()` or `getHaltReason()`. Otherwise call `abort()` and discard it.
  A resolved `processStream()` alone does not mean success.
- Calls execute in the graph by default. To hand a complete call to the OpenAI
  client instead, set `clientDelegatedToolNames: ['my_tool']` in `Run.create`
  for a single-agent run and register its model-facing schema. The graph ends
  without executing that call. Batches mixing delegated and graph/provider
  tools fail closed; make separate model turns. ToolNode claims remain an
  additional guard, never proof that an unclaimed call belongs to the client.
- `emit` is synchronous. Failed/partial writes cannot be retried; the host owns
  HTTP backpressure. This helper does not provide durable resume or undo tool effects.

Projected arguments must be JSON data: primitives, plain objects and dense arrays.
Getters, custom objects, cycles and nesting beyond 64 levels are rejected before projection.
Observers receive isolated snapshots; provider IDs are reserved before synthetic IDs.
Accepted-event snapshots cap each response at **1,024 calls / 4 MiB**. Projection
uses those defaults for pending calls (`maxToolCalls` / `maxBufferedBytes`). These
formatting limits do not apply to ordinary runs without accepted-event handlers;
descriptor safety checks still apply before stream accounting and dispatch.
These are output limits, not bounds on provider memory or tool execution.

See [the design decision](docs/adr/0010-project-accepted-model-results.md).

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

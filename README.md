# @librechat/agents

TypeScript utilities for building LibreChat agent workflows. The package provides graph orchestration, streaming event handling, tool execution, provider adapters, and message formatting for single-agent and multi-agent runs.

## Features

- LangGraph-based single-agent and multi-agent workflows
- Streaming content aggregation and run-step event handlers
- Tool calling, tool search, subagent handoffs, and programmatic tool execution
- Provider adapters for Anthropic, Bedrock, Vertex AI, OpenAI-compatible providers, Google, Mistral, DeepSeek, and xAI
- Message formatting, context pruning, summarization, and cache-control helpers
- Native Google image output through a [host-supplied media port](docs/native-media.md)

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

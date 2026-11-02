# Serialized request body budgets

OpenAI and Azure clients accept an optional `maxRequestBodyBytes` client option.
It limits UTF-8 bytes in the SDK-built body passed to the configured transport,
after JSON serialization, cache markers and request-option overrides. Token
budgets remain separate. The option is unset by default.

```ts
import { Providers, Run } from '@librechat/agents';

const run = await Run.create({
  graphConfig: {
    type: 'standard',
    maxContextTokens: 16_000,
    llmConfig: {
      provider: Providers.OPENAI,
      model: 'your-model',
      useResponsesApi: true,
      maxTokens: 4_096,
      maxRequestBodyBytes: 131_072,
    },
  },
  tokenCounter,
});
```

Supply the usual credentials and token counter through the host application.
Choose the byte limit to match that application's transport or gateway. This
option does not change gateway policy, pricing, output limits or monetary caps.
Each fallback client needs its own configured limit.

## Admission and recovery

The custom SDK client's `buildRequest` hook checks the actual serialized string.
It does not serialize it again or change a passing body. A rejected body never
reaches the configured fetch. `RequestBodyLimitError` contains only measured
bytes, configured bytes and a local source marker; it is not a provider context
overflow. Both SDK transport retries and LangChain retries are bypassed, including
when a host supplies its own retry handler.

In the graph, a primary size rejection can reduce the context budget to 70% of
its previous value and rebuild the provider projection from canonical history.
This is a heuristic compaction actuator, not a conversion from bytes to tokens.
The retry is serialized and measured again. No provider calibration is inferred
from the byte count.

Recovery shares the existing allowance of two forced compactions per agent per
run. It stops sooner if the same history's body does not shrink, a token counter
or context budget is missing, or the reduced budget lacks 2,000 message tokens
after instructions and the normal reserve. Reset restores the original budget
and recovery allowance. Canonical messages are retained; existing projection
rules still determine which history reaches the provider.

A `request-body` routing detour performs no summary model call. Ordinary summary
triggers are also suppressed for the unchanged history being retried. New
messages may trigger normal configured summarization later.

If a primary body remains inadmissible, its local error propagates before
fallback selection. If an independently selected fallback hits its own local
body limit, the remaining fallback chain stops. The graph does not rerun a paid
primary merely because a fallback's body failed locally. Other provider failures
retain their existing fallback behavior.

## Supported scope

- Built-in LibreChat OpenAI and Azure clients, Chat Completions and Responses,
  including streaming and structured-output parsing.
- SDK-built string bodies, including effective `RequestOptions.body` and
  `fetchOptions.body` overrides. Non-string bodies fail locally without being
  consumed.
- Replacement request delegates and custom SDK provider hooks are rejected when
  the limit is enabled because they can bypass or mutate SDK serialization.
- A custom fetch can independently rewrite its outbound requests. The limit
  constrains the SDK-built body passed to that fetch, not arbitrary code inside
  the transport.
- Anthropic Messages and other provider wrappers do not use this option. Keep
  independent gateway admission controls in place.

This guard prevents a class of local transport denials and permits bounded
recovery. It does not establish better coding-task quality, speed or cost. In
particular, a smaller projection can omit useful context, and irreducible
instructions, schemas, arguments or opaque state may still exceed the limit.
The failed scale attempt that motivated this work remains failed and unpaired.
